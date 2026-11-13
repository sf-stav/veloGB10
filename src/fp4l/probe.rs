//! `--probe-fp4l --model-dir <FP4-Local dir> [--fp4l-ref <bf16 dir>] [--fp4l-threads N]` — CPU-only
//! receipts for the FP4-Local pack (FP4L-0). No GPU, no model load.
//!
//! 1. Census: every linear classified by the tensors next to it; cross-checked against
//!    `quantization_config.quantized_layers`.
//! 2. csf: every compressed scale plane decoded, re-encoded with the producer's rule and compared
//!    byte-for-byte with the stored streams (FP4L_CSF_ROUNDTRIP).
//! 3. MXFP8: every dequantized value checked for exact f16 representability; the scale exponent
//!    range (FP4L_MX8_F16).
//! 4. NVFP4: scale bytes finite and positive, `weight_scale_2` finite and positive.
//! 5. With `--fp4l-ref` (a bf16 Qwen3.8-Flash-Next or the 4-layer slice): per-class rel-L2 / cosine
//!    of the dequantized FP4-Local tensors against the bf16 source, the expert gate|up order, PLE
//!    rows, and exact equality of the bf16 classes (FP4L_REF).

use super::pack::{self, Fp4lPack, LinKind, StDir};
use super::csf;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

fn threads(n: Option<usize>) -> usize {
    n.unwrap_or_else(|| std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8)).clamp(1, 64)
}

/// Comparison accumulators (f64): sum (a-b)^2, sum b^2, sum a*b, sum a^2, max |a-b|.
#[derive(Default, Clone, Copy)]
struct Cmp { d2: f64, b2: f64, ab: f64, a2: f64, maxd: f64, n: u64, neq: u64 }
impl Cmp {
    fn add(&mut self, a: &[f32], b: &[f32]) {
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            self.d2 += (x - y) * (x - y); self.b2 += y * y; self.ab += x * y; self.a2 += x * x;
            self.maxd = self.maxd.max((x - y).abs()); self.n += 1;
            if x != y { self.neq += 1; }
        }
    }
    fn merge(&mut self, o: &Cmp) { self.d2 += o.d2; self.b2 += o.b2; self.ab += o.ab; self.a2 += o.a2; self.maxd = self.maxd.max(o.maxd); self.n += o.n; self.neq += o.neq; }
    fn rel(&self) -> f64 { (self.d2 / self.b2.max(1e-300)).sqrt() }
    fn cos(&self) -> f64 { self.ab / (self.a2.sqrt() * self.b2.sqrt()).max(1e-300) }
    fn line(&self) -> String { format!("rel-L2 {:.4} cos {:.6} max|d| {:.3e} differ {}/{}", self.rel(), self.cos(), self.maxd, self.neq, self.n) }
}

/// Every linear stem (name without `.weight`) with its kind.
pub fn linear_stems(p: &Fp4lPack) -> Result<BTreeMap<String, LinKind>> {
    let mut out = BTreeMap::new();
    for (name, r) in &p.t {
        let Some(stem) = name.strip_suffix(".weight") else { continue };
        if r.shape.len() != 2 { continue; }
        // skip norms / embeddings / PLE tables stored as `.weight` without being linears we serve as GEMMs
        if stem.ends_with("embed_tokens") || stem.contains("ngram_embedding") { continue; }
        match p.lin_kind(stem) {
            Ok(k) => { out.insert(stem.to_string(), k); }
            Err(_) => {}
        }
    }
    Ok(out)
}

/// The main.rs dispatch of the two FP4L probes (both listed by their registry lines in --help-advanced):
/// `--probe-fp4l --model-dir DIR [--fp4l-ref BF16_DIR] [--fp4l-threads N]` = the CPU receipts (census, csf round
/// trip, MXFP8 f16 exactness, NVFP4 sanity, optional bf16 reference comparison); `--probe-fp4l-gpu --model-dir DIR`
/// = the FP4L kernels on real pack tensors vs an exact host reference. true = a probe ran (the caller returns);
/// a failing probe exits 1 with its FAIL line.
pub fn dispatch(args: &[String]) -> bool {
    if args.iter().any(|a| a == "--probe-fp4l") {
        let dir = crate::arg_value(args, "--model-dir").expect("--probe-fp4l requires --model-dir <FP4-Local dir>");
        let rf = crate::arg_value(args, "--fp4l-ref");
        let nt: Option<usize> = crate::arg_value(args, "--fp4l-threads").and_then(|s| s.parse().ok());
        probe_fp4l(dir, rf, nt).unwrap_or_else(|e| {
            eprintln!("FP4L_PROBE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return true;
    }
    if args.iter().any(|a| a == "--probe-fp4l-gpu") {
        let dir = crate::arg_value(args, "--model-dir").expect("--probe-fp4l-gpu requires --model-dir <FP4-Local dir>");
        crate::exl3_forward::probe_fp4l_gpu(dir).unwrap_or_else(|e| {
            eprintln!("FP4L_GPU_FAIL: {e:#}");
            std::process::exit(1);
        });
        return true;
    }
    false
}

pub fn probe_fp4l(dir: &str, ref_dir: Option<&str>, nthreads: Option<usize>) -> Result<()> {
    let t0 = std::time::Instant::now();
    let p = Fp4lPack::open(Path::new(dir)).context("FP4L pack open")?;
    let nt = threads(nthreads);
    println!("FP4L probe: {dir}: {} tensors in {} files, header read {:.1} s, {} threads",
             p.t.len(), p.files.len(), t0.elapsed().as_secs_f64(), nt);

    // ---------------------------------------------------------------- 1. census
    let stems = linear_stems(&p)?;
    let mut by_kind: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut by_class: BTreeMap<String, (LinKind, usize)> = BTreeMap::new();
    for (stem, k) in &stems {
        let (n, kk) = p.shape2(&format!("{stem}.weight"))?;
        let kdim = if matches!(k, LinKind::Nvfp4 | LinKind::Nvfp4Csf) { kk * 2 } else { kk };
        let e = by_kind.entry(format!("{k:?}")).or_default();
        e.0 += 1; e.1 += (n * kdim) as u64;
        let class = stem.split('.').map(|s| if s.chars().all(|c| c.is_ascii_digit()) { "N" } else { s }).collect::<Vec<_>>().join(".");
        let c = by_class.entry(class).or_insert((*k, 0));
        anyhow::ensure!(c.0 == *k, "FP4L census: class of {stem} mixes {:?} and {k:?}", c.0);
        c.1 += 1;
    }
    for (k, (n, params)) in &by_kind { println!("  census {k:<9} {n:>6} linears {:>14} params", params); }
    for (c, (k, n)) in &by_class { println!("    {n:>6} x {k:?}  {c}"); }
    // config cross-check: each quantized_layers entry names a module or a module group (`...experts`)
    let mut xbad = 0usize;
    for (module, algo) in &p.qlayers {
        let hits: Vec<&LinKind> = stems.iter().filter(|(s, _)| *s == module || s.starts_with(&format!("{module}."))).map(|(_, k)| k).collect();
        let want = |k: &LinKind| match algo.as_str() {
            "MXFP8" => *k == LinKind::Mxfp8,
            "NVFP4" => matches!(k, LinKind::Nvfp4Csf | LinKind::Nvfp4),
            "W4A16_NVFP4" => *k == LinKind::Nvfp4,
            _ => false,
        };
        let ple = module.contains("ngram_embedding");
        if hits.is_empty() && !ple {
            // the config's MTP experts key is `mtp.layers.48.*` while the tensors are `mtp.layers.0.*`
            let alt = module.replace("mtp.layers.48", "mtp.layers.0");
            let alt_hits = stems.iter().filter(|(s, _)| s.starts_with(&format!("{alt}."))).count();
            if alt_hits == 0 { xbad += 1; println!("    CONFIG-XCHECK: {module} ({algo}) matches no tensor"); }
            else { println!("    config names {module}; tensors are under {alt} ({alt_hits} linears) — accepted"); }
        } else if !hits.iter().all(|k| want(k)) {
            xbad += 1;
            println!("    CONFIG-XCHECK: {module} says {algo}, tensors are {:?}", hits.iter().take(3).collect::<Vec<_>>());
        }
    }
    println!("FP4L_CENSUS {} linears, config cross-check mismatches {xbad}", stems.len());

    // ---------------------------------------------------------------- 2. csf round trip
    let csf_stems: Vec<&String> = stems.iter().filter(|(_, k)| **k == LinKind::Nvfp4Csf).map(|(s, _)| s).collect();
    let t1 = std::time::Instant::now();
    let done = AtomicUsize::new(0);
    let bad = AtomicUsize::new(0);
    let exc_words = AtomicU64::new(0);
    let plane_bytes = AtomicU64::new(0);
    let stream_bytes = AtomicU64::new(0);
    let first_bad: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    std::thread::scope(|sc| {
        for t in 0..nt {
            let (csf_stems, p, done, bad, exc_words, plane_bytes, stream_bytes, first_bad) =
                (&csf_stems, &p, &done, &bad, &exc_words, &plane_bytes, &stream_bytes, &first_bad);
            sc.spawn(move || {
                let mut i = t;
                while i < csf_stems.len() {
                    let stem = csf_stems[i];
                    let r = (|| -> Result<bool> {
                        let fname = format!("{stem}.weight_scale.nvfp4_csf_fixed");
                        let ename = format!("{stem}.weight_scale.nvfp4_csf_exceptions");
                        let shape = p.get(&fname)?.shape.clone();
                        let fixed = p.raw(&fname)?;
                        let exc = p.raw(&ename)?;
                        let (rows, cols) = csf::geometry(&shape)?;
                        let plane = csf::decode(&fixed, &shape, &exc)?;
                        let (f2, e2) = csf::encode(&plane, rows, cols)?;
                        exc_words.fetch_add(exc.len() as u64 / 4, Ordering::Relaxed);
                        plane_bytes.fetch_add(plane.len() as u64, Ordering::Relaxed);
                        stream_bytes.fetch_add((fixed.len() + exc.len()) as u64, Ordering::Relaxed);
                        Ok(f2 == fixed && e2 == exc && plane.iter().all(|&b| b & 0x80 == 0 && b & 0x7F != 0x7F))
                    })();
                    match r {
                        Ok(true) => {}
                        Ok(false) => { bad.fetch_add(1, Ordering::Relaxed); first_bad.lock().unwrap().get_or_insert(stem.clone()); }
                        Err(e) => { bad.fetch_add(1, Ordering::Relaxed); first_bad.lock().unwrap().get_or_insert(format!("{stem}: {e:#}")); }
                    }
                    done.fetch_add(1, Ordering::Relaxed);
                    i += nt;
                }
            });
        }
    });
    let nb = bad.load(Ordering::Relaxed);
    println!("FP4L_CSF_ROUNDTRIP {} planes={} bad={} exceptions={} decoded={:.2} GB from {:.2} GB of streams ({:.1}%) in {:.1} s{}",
             if nb == 0 && !csf_stems.is_empty() { "OK" } else { "FAIL" }, done.load(Ordering::Relaxed), nb,
             exc_words.load(Ordering::Relaxed), plane_bytes.load(Ordering::Relaxed) as f64 / 1e9,
             stream_bytes.load(Ordering::Relaxed) as f64 / 1e9,
             100.0 * stream_bytes.load(Ordering::Relaxed) as f64 / plane_bytes.load(Ordering::Relaxed).max(1) as f64,
             t1.elapsed().as_secs_f64(),
             first_bad.lock().unwrap().as_ref().map(|s| format!(" first: {s}")).unwrap_or_default());

    // ---------------------------------------------------------------- 3. MXFP8 f16 exactness
    let mx_stems: Vec<&String> = stems.iter().filter(|(_, k)| **k == LinKind::Mxfp8).map(|(s, _)| s).collect();
    let t2 = std::time::Instant::now();
    let res: std::sync::Mutex<Vec<(String, u64, u64, u64, i32, i32, u64)>> = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for t in 0..nt {
            let (mx_stems, p, res) = (&mx_stems, &p, &res);
            sc.spawn(move || {
                let mut i = t;
                while i < mx_stems.len() {
                    let stem = mx_stems[i];
                    if let Ok(m) = p.mxfp8(stem) {
                        let (mut inexact, mut nan, mut zero) = (0u64, 0u64, 0u64);
                        let (mut emin, mut emax) = (i32::MAX, i32::MIN);
                        let mut sub = 0u64; // values that land in the f16 SUBNORMAL range (exact or not)
                        for r in 0..m.n {
                            for g in 0..m.k / 32 {
                                let sb = m.scales[r * (m.k / 32) + g];
                                if sb == 0xFF { nan += 32; continue; }
                                let e = sb as i32 - 127;
                                emin = emin.min(e); emax = emax.max(e);
                                let s = pack::ue8m0(sb);
                                for c in 0..32 {
                                    let code = m.codes[r * m.k + g * 32 + c];
                                    if code & 0x7F == 0x7F { nan += 1; continue; }
                                    let v = pack::e4m3(code) * s;
                                    if v == 0.0 { zero += 1; continue; }
                                    if v.abs() < 6.103_515_6e-5 { sub += 1; }
                                    if !pack::f16_exact(v) { inexact += 1; }
                                }
                            }
                        }
                        res.lock().unwrap().push((stem.clone(), inexact, nan, zero, emin, emax, sub));
                    } else {
                        res.lock().unwrap().push((stem.clone(), u64::MAX, 0, 0, 0, 0, 0));
                    }
                    i += nt;
                }
            });
        }
    });
    let res = res.into_inner().unwrap();
    let mut cls: BTreeMap<String, (u64, u64, i32, i32, u64, usize)> = BTreeMap::new();
    for (stem, inex, nan, _z, emin, emax, sub) in &res {
        let class = stem.split('.').map(|s| if s.chars().all(|c| c.is_ascii_digit()) { "N" } else { s }).collect::<Vec<_>>().join(".");
        let c = cls.entry(class).or_insert((0, 0, i32::MAX, i32::MIN, 0, 0));
        c.0 = c.0.saturating_add(*inex); c.1 += nan; c.2 = c.2.min(*emin); c.3 = c.3.max(*emax); c.4 += sub; c.5 += 1;
    }
    let tot_inex: u64 = res.iter().map(|r| r.1).fold(0u64, |a, b| a.saturating_add(b));
    let tot_nan: u64 = res.iter().map(|r| r.2).sum();
    for (c, (inex, nan, emin, emax, sub, n)) in &cls {
        println!("    mxfp8 {n:>4} x {c}: scale 2^[{emin}, {emax}], f16-subnormal values {sub}, f16-inexact {inex}, NaN {nan}");
    }
    println!("FP4L_MX8_F16 {} tensors={} f16-inexact={} NaN={} in {:.1} s (inexact>0 only matters for an f16 weight upconvert; the W8A16 kernels scale the fp32 accumulator per 32-group)",
             if tot_nan == 0 { "OK" } else { "FAIL" }, res.len(), tot_inex, tot_nan, t2.elapsed().as_secs_f64());

    // ---------------------------------------------------------------- 4. NVFP4 (plain-scale) sanity
    let nv_stems: Vec<&String> = stems.iter().filter(|(_, k)| **k == LinKind::Nvfp4).map(|(s, _)| s).collect();
    let mut nv_bad = 0usize;
    let (mut gmin, mut gmax) = (f32::MAX, 0f32);
    for stem in &nv_stems {
        let h = p.nvfp4(stem)?;
        if h.scales.iter().any(|&b| b & 0x80 != 0 || b & 0x7F == 0x7F) { nv_bad += 1; }
        gmin = gmin.min(h.gs); gmax = gmax.max(h.gs);
    }
    println!("FP4L_NV4_PLAIN {} tensors={} bad-scale tensors={} weight_scale_2 in [{gmin:.3e}, {gmax:.3e}]",
             if nv_bad == 0 { "OK" } else { "FAIL" }, nv_stems.len(), nv_bad);

    // ---------------------------------------------------------------- 4b. the precision ledger, per lever (design §3a)
    // How the engine holds every unquantized tensor of the pack, and what the f16 end of each lever rounds:
    //   f32-held by the trunk at every setting (GDN conv1d / A_log / dt_bias): exact;
    //   --fp4l-plain   trunk-held bf16 (hc mixers + inject, router, shared-expert gate, embedding, n-gram key /
    //                  value / conv): exact = the stored bits; f16 = the trunk's f16 copies;
    //   --fp4l-norms   the fp32 norms (hc_norm, n-gram norms) + the MTP layer's bf16 hc / pre-fc norms: f32 = stored;
    //   --fp4l-lm-head lm_head: bf16 = stored; f16 = one 2^k shift per tensor; mxfp8 = RTN (lossy by design);
    //   MTP linears    (follow --fp4l-plain): exact = stored; f16 = one 2^k shift per tensor;
    //   other          (attention / GDN / indexer norms): held in f16 at every setting — must be exact (checked).
    {
        #[derive(Default)]
        struct Row { n_t: usize, vals: u64, inexact: u64, max_bad: f32 }
        let mut led: BTreeMap<&'static str, Row> = BTreeMap::new();
        let mtp_lin = |n: &str| n == "lm_head.weight" || n.starts_with("mtp.fc_")
            || (n.starts_with("mtp.layers.0.self_attn.") && n.ends_with("_proj.weight"))
            || n.starts_with("mtp.layers.0.mlp.shared_expert.");
        let class_of = |n: &str, dt: &pack::Dt| -> &'static str {
            if n == "lm_head.weight" { return "lm_head (--fp4l-lm-head)"; }
            if mtp_lin(n) { return "MTP linears (--fp4l-plain)"; }
            if n.ends_with("linear_attn.conv1d.weight") || n.ends_with("linear_attn.A_log") || n.ends_with("linear_attn.dt_bias") {
                return "f32-held (GDN conv / A_log / dt_bias)";
            }
            if n.ends_with("weight_scale_2") || n.ends_with("input_scale") || n.ends_with("weight_scale") {
                return "fp32 scale factors (kept fp32)";
            }
            if (matches!(dt, pack::Dt::F32) && n.contains("norm")) || n.ends_with("hc_norm.weight") || n.starts_with("mtp.pre_fc_norm_") {
                return "norms (--fp4l-norms)";
            }
            if n.contains("input_mix_weight_") || n.contains("block_inject_weight") || n.ends_with("mlp.gate.weight")
                || n.ends_with("shared_expert_gate.weight") || n.ends_with("embed_tokens.weight")
                || n.ends_with("ple.key_proj.weight") || n.ends_with("ple.value_proj.weight") || n.ends_with("ple.conv1d.weight") {
                return "trunk-held bf16 (--fp4l-plain)";
            }
            if matches!(dt, pack::Dt::F32) { return "other fp32 (unclassified)"; }
            "other f16-held"
        };
        let names: Vec<&String> = p.t.iter().filter(|(n, r)| matches!(r.dtype, pack::Dt::Bf16 | pack::Dt::F32)
            && !n.starts_with("model.visual.") && !n.contains("ngram_embedding")).map(|(n, _)| n).collect();
        for n in names {
            let dt = p.get(n)?.dtype.clone();
            let c = class_of(n, &dt);
            let v = p.plain_f32(n)?;
            let bad: Vec<f32> = match c {
                "f32-held (GDN conv / A_log / dt_bias)" | "fp32 scale factors (kept fp32)" => Vec::new(),
                "lm_head (--fp4l-lm-head)" | "MTP linears (--fp4l-plain)" => {
                    let sc = 2f32.powi(pack::exact_f16_shift(&v));
                    v.iter().copied().filter(|&x| !pack::f16_exact(x * sc)).collect()
                }
                _ => v.iter().copied().filter(|x| !pack::f16_exact(*x)).collect(),
            };
            let r = led.entry(c).or_default();
            r.n_t += 1; r.vals += v.len() as u64; r.inexact += bad.len() as u64;
            r.max_bad = bad.iter().fold(r.max_bad, |a, x| a.max(x.abs()));
        }
        for (c, r) in &led {
            println!("    {c:<40} {:>4} tensors {:>13} values: {:>9} not exact at the lever's f16 end (largest |v| {:.2e}){}",
                     r.n_t, r.vals, r.inexact, r.max_bad,
                     if *c == "other f16-held" && r.inexact > 0 { "  <-- held in f16 at EVERY setting: a gap" } else { "" });
        }
        let other_bad = led.get("other f16-held").map_or(0, |r| r.inexact);
        let f16_end: u64 = led.iter().filter(|(c, _)| c.contains("--fp4l-")).map(|(_, r)| r.inexact).sum();
        println!("FP4L_LEDGER {} full-precision ends (--fp4l-plain exact --fp4l-norms f32 --fp4l-lm-head bf16): 0 rounded values; \
                  f16 ends: {f16_end} rounded; held in f16 at every setting: {other_bad} rounded",
                 if other_bad == 0 { "EXACT" } else { "GAP" });
    }

    // MTP head inventory
    let mut mtp_layers: Vec<String> = p.t.keys().filter_map(|k| k.strip_prefix("mtp.layers.").and_then(|r| r.split('.').next()).map(String::from)).collect();
    mtp_layers.sort(); mtp_layers.dedup();
    println!("  mtp: tensor layer indices {:?}; config mtp_num_hidden_layers {:?}", mtp_layers,
             p.text_config().get("mtp_num_hidden_layers"));

    // ---------------------------------------------------------------- engine config + fit
    let cfg = crate::qwen::Config::from_config_json(&format!("{dir}/config.json"))
        .context("FP4L: the engine's qwen::Config parse of config.json")?;
    println!("  engine config: family {:?}, {} layers, hidden {}, heads {}/{} x {}, experts {} top-{} mi {}, shared {}, hc {}x{}, \
              indexer {}x{} budget {}, vocab {}, eos {}", cfg.family, cfg.num_layers, cfg.hidden_size,
             cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
             cfg.shared_expert_intermediate_size, cfg.hc_count, cfg.hc_lowrank, cfg.indexer_n_heads, cfg.indexer_head_dim,
             cfg.indexer_budget, cfg.vocab_size, cfg.eos_token_id);
    // the EXL3 pack this trunk already serves is the structural reference: every field must agree
    for refp in ["Qwen3.8-Flash-Next-exl3-3.05bpw", "Qwen3.8-Flash-Next-EXL3-4.05bpw"] {
        let rp = Path::new(dir).parent().map(|d| d.join(refp)).filter(|d| d.join("config.json").is_file());
        if let Some(rp) = rp {
            if let Ok(rc) = crate::qwen::Config::from_config_json(&rp.join("config.json").to_string_lossy()) {
                let (a, b) = (format!("{cfg:?}"), format!("{rc:?}"));
                let da: Vec<&str> = a.split(", ").collect();
                let db: Vec<&str> = b.split(", ").collect();
                let diffs: Vec<String> = da.iter().zip(db.iter()).filter(|(x, y)| x != y).map(|(x, y)| format!("{x} | exl3: {y}")).collect();
                println!("FP4L_CFG_VS_EXL3 {} fields compared with {refp}: {} differ{}", da.len(), diffs.len(),
                         if diffs.is_empty() { String::new() } else { format!(" -> {}", diffs.join(" ; ")) });
                break;
            }
        }
    }
    if let Some((d, t, tw)) = crate::fp4l::fit_bytes(Path::new(dir)) {
        println!("  fit: device weights {:.2} GB (csf scales decoded), n-gram table {:.2} GB (RAM/SSD), tower f32 {:.2} GB (head only)",
                 d as f64 / 1e9, t as f64 / 1e9, tw as f64 / 1e9);
    }

    // ---------------------------------------------------------------- 5. reference comparison
    if let Some(rd) = ref_dir {
        compare_ref(&p, &stems, rd, nt)?;
    }
    println!("FP4L probe done in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Dequantized rows `[r0, r1)` of an FP4L linear as f32 (`stem` without `.weight`).
fn fp4l_rows(p: &Fp4lPack, stem: &str, kind: LinKind, r0: usize, r1: usize) -> Result<(Vec<f32>, usize)> {
    match kind {
        LinKind::Mxfp8 => {
            let m = p.mxfp8_rows(stem, r0, r1)?;
            let mut out = vec![0f32; m.n * m.k];
            for r in 0..m.n { m.row_f32(r, &mut out[r * m.k..(r + 1) * m.k]); }
            Ok((out, m.k))
        }
        LinKind::Nvfp4 | LinKind::Nvfp4Csf => {
            let h = p.nvfp4(stem)?;
            let mut out = vec![0f32; (r1 - r0) * h.k];
            for r in r0..r1 { h.row_f32(r, &mut out[(r - r0) * h.k..(r - r0 + 1) * h.k]); }
            Ok((out, h.k))
        }
        LinKind::Bf16 => {
            let (_n, k) = p.shape2(&format!("{stem}.weight"))?;
            Ok((p.plain_f32_range(&format!("{stem}.weight"), r0 * k, r1 * k)?, k))
        }
    }
}

fn compare_ref(p: &Fp4lPack, stems: &BTreeMap<String, LinKind>, rd: &str, nt: usize) -> Result<()> {
    let r = StDir::open(Path::new(rd)).with_context(|| format!("open reference {rd}"))?;
    println!("  reference {rd}: {} tensors", r.t.len());
    let mut agg: BTreeMap<String, Cmp> = BTreeMap::new();
    // dense linears present in both under the same name
    for (stem, kind) in stems {
        let wn = format!("{stem}.weight");
        if !r.has(&wn) || stem.contains(".experts.") { continue; }
        let (n, _) = p.shape2(&wn)?;
        let (a, k) = fp4l_rows(p, stem, *kind, 0, n)?;
        anyhow::ensure!(r.numel(&wn)? == n * k, "ref {wn} numel {} != {}", r.numel(&wn)?, n * k);
        let b = r.plain_f32(&wn)?;
        let class = format!("{kind:?} {}", stem.split('.').map(|s| if s.chars().all(|c| c.is_ascii_digit()) { "N" } else { s }).collect::<Vec<_>>().join("."));
        let mut c = Cmp::default(); c.add(&a, &b);
        agg.entry(class).or_default().merge(&c);
    }
    // routed experts: the ref stacks [E, 2*I, H] gate_up_proj and [E, H, I] down_proj
    for li in 0..64usize {
        let gu = format!("model.language_model.layers.{li}.mlp.experts.gate_up_proj");
        let dn = format!("model.language_model.layers.{li}.mlp.experts.down_proj");
        if !r.has(&gu) { continue; }
        let sh = r.get(&gu)?.shape.clone();
        let (ne, i2, hdim) = (sh[0], sh[1], sh[2]);
        let im = i2 / 2;
        let experts: Vec<usize> = if li == 0 { (0..ne).collect() } else { vec![0, 1, ne / 2, ne - 1] };
        let cm: std::sync::Mutex<[Cmp; 5]> = std::sync::Mutex::new([Cmp::default(); 5]);
        std::thread::scope(|sc| {
            for t in 0..nt {
                let (experts, cm, p, r, gu, dn) = (&experts, &cm, p, &r, &gu, &dn);
                sc.spawn(move || {
                    let mut i = t;
                    while i < experts.len() {
                        let e = experts[i];
                        let res = (|| -> Result<[Cmp; 5]> {
                            let base = format!("model.language_model.layers.{li}.mlp.experts.{e}");
                            let (g, _) = fp4l_rows(p, &format!("{base}.gate_proj"), LinKind::Nvfp4Csf, 0, im)?;
                            let (u, _) = fp4l_rows(p, &format!("{base}.up_proj"), LinKind::Nvfp4Csf, 0, im)?;
                            let (d, _) = fp4l_rows(p, &format!("{base}.down_proj"), LinKind::Nvfp4Csf, 0, hdim)?;
                            let rg = r.plain_f32_range(gu, e * i2 * hdim, e * i2 * hdim + im * hdim)?;
                            let ru = r.plain_f32_range(gu, e * i2 * hdim + im * hdim, (e + 1) * i2 * hdim)?;
                            let rdn = r.plain_f32_range(dn, e * hdim * im, (e + 1) * hdim * im)?;
                            let mut o = [Cmp::default(); 5];
                            o[0].add(&g, &rg); o[1].add(&u, &ru); o[2].add(&d, &rdn);
                            o[3].add(&g, &ru); o[4].add(&u, &rg); // swapped order (must be far worse)
                            Ok(o)
                        })();
                        match res {
                            Ok(o) => { let mut a = cm.lock().unwrap(); for j in 0..5 { a[j].merge(&o[j]); } }
                            Err(e) => eprintln!("    expert compare L{li} e{e}: {e:#}"),
                        }
                        i += nt;
                    }
                });
            }
        });
        let a = cm.into_inner().unwrap();
        println!("    L{li} experts ({} of {ne}): gate {} | up {} | down {}", experts.len(), a[0].line(), a[1].line(), a[2].line());
        println!("    L{li} experts swapped-order control: gate-vs-up rel-L2 {:.3}, up-vs-gate {:.3} (gate-first order holds iff these are ~1.4)", a[3].rel(), a[4].rel());
        agg.entry("NVFP4 experts.gate".into()).or_default().merge(&a[0]);
        agg.entry("NVFP4 experts.up".into()).or_default().merge(&a[1]);
        agg.entry("NVFP4 experts.down".into()).or_default().merge(&a[2]);
    }
    // plain (unquantized in FP4-Local) tensors: compare everything 1-D / small, and the big bf16 ones
    let mut plain_eq: BTreeMap<String, Cmp> = BTreeMap::new();
    for (name, tr) in &p.t {
        if !r.has(name) || stems.contains_key(name.trim_end_matches(".weight")) && name.ends_with(".weight") && tr.dtype != pack::Dt::Bf16 { continue; }
        if !matches!(tr.dtype, pack::Dt::Bf16 | pack::Dt::F32) { continue; }
        if name.contains("ngram_embedding") { continue; }
        let a = p.plain_f32(name)?;
        let b = r.plain_f32(name)?;
        if a.len() != b.len() { println!("    shape differs: {name} {} vs {}", a.len(), b.len()); continue; }
        let class = format!("{:?} {}", tr.dtype, name.split('.').map(|s| if s.chars().all(|c| c.is_ascii_digit()) { "N" } else { s }).collect::<Vec<_>>().join("."));
        let mut c = Cmp::default(); c.add(&a, &b);
        plain_eq.entry(class).or_default().merge(&c);
    }
    // PLE table: sample rows of a few shards
    let mut ple = Cmp::default();
    let ple_stem = p.t.keys().find(|k| k.ends_with("ngram_embedding.shard_0.weight")).cloned();
    if let Some(ws0) = ple_stem {
        let pre = ws0.trim_end_matches("shard_0.weight").to_string();
        let gs = p.scalar_f32(&format!("{pre}weight_scale_2"))?;
        for s in [0usize, 1, 63, 127] {
            let wn = format!("{pre}shard_{s}.weight");
            let sn = format!("{pre}shard_{s}.weight_scale");
            let rn = format!("{pre}shard_{s}.weight");
            if !r.has(&rn) { continue; }
            let rows = p.get(&wn)?.shape[0];
            let dim = p.get(&wn)?.shape[1] * 2;
            for start in [0usize, rows / 3, rows - 512] {
                let q = p.raw_range(&wn, (start * dim / 2) as u64, ((start + 512) * dim / 2) as u64)?;
                let sc = p.raw_range(&sn, (start * dim / 16) as u64, ((start + 512) * dim / 16) as u64)?;
                let h = pack::Nv4Host { packed: q, scales: sc, gs, n: 512, k: dim };
                let mut a = vec![0f32; 512 * dim];
                for rr in 0..512 { h.row_f32(rr, &mut a[rr * dim..(rr + 1) * dim]); }
                let b = r.plain_f32_range(&rn, start * dim, (start + 512) * dim)?;
                ple.add(&a, &b);
            }
        }
        println!("    PLE rows (4 shards x 1536 rows, NVFP4 x weight_scale_2 {gs:.4e}): {}", ple.line());
    }
    for (c, v) in &agg { println!("  REF {c:<60} {}", v.line()); }
    for (c, v) in &plain_eq { println!("  REF-PLAIN {c:<55} {}", v.line()); }
    let worst = agg.values().map(|c| c.cos()).fold(1.0f64, f64::min);
    println!("FP4L_REF {} classes={} worst cosine {worst:.5} (layout errors show up as cosine << 0.9; QAD moved the weights, so equality is not expected)",
             if worst > 0.9 { "OK" } else { "SUSPECT" }, agg.len());
    Ok(())
}
