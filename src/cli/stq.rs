//! STQ bake and its check / verify modes. Sibling of `quantize.rs`.

use super::*;

/// SQ campaign probe bake: dequant NVFP4 -> STQ/ternary2/3bit round-trip -> requant NVFP4,
/// values-only (format/shards/names unchanged, served by the current binary). Mirrors
/// run_requant_sim's shape (shard resume, triple handling, aux-file copy).
///   --stq-bake --model-dir <packed-dir> --out <dir> --arm a|b [--imatrix <file>]
///              [--classes gateup,down,attn] [--shard-start N] [--shard-end N] [--limit N]
///              [--check]  (check = fitter unit gate: LS-vs-amax SSD on real rows, no bake)
pub(crate) fn run_stq_bake(args: &[String]) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    use gb10_inference::quant;

    if args.iter().any(|a| a == "--check") {
        stq_check_mode(args);
        return;
    }
    if args.iter().any(|a| a == "--verify") {
        stq_verify_mode(args);
        return;
    }
    let ind_s = parse_arg(args, "--model-dir").expect("--stq-bake requires --model-dir <packed-dir>");
    let out_s = parse_arg(args, "--out").expect("--stq-bake requires --out <dir>");
    let arm = parse_arg(args, "--arm").unwrap_or("a");
    assert!(arm == "a" || arm == "b" || arm == "c" || arm == "d" || arm == "e", "--arm must be a-e");
    let classes: Vec<&str> = parse_arg(args, "--classes").unwrap_or("gateup,down,attn")
        .split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    let shard_start: usize = parse_arg(args, "--shard-start").and_then(|s| s.parse().ok()).unwrap_or(1);
    let shard_end: usize = parse_arg(args, "--shard-end").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let limit: usize = parse_arg(args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let ind = std::path::Path::new(ind_s);
    let outd = std::path::Path::new(out_s);
    std::fs::create_dir_all(outd).expect("create --out dir");

    // imatrix: DATA input (per-input-channel importance, key = tensor stem, f32 [K]).
    let imatrix: Option<std::collections::HashMap<String, Vec<f32>>> = parse_arg(args, "--imatrix").map(|p| {
        let raw = std::fs::read(&p).unwrap_or_else(|e| panic!("read imatrix {p}: {e}"));
        let st = SafeTensors::deserialize(&raw).expect("parse imatrix");
        let mut m = std::collections::HashMap::new();
        for (n, v) in st.tensors() {
            assert_eq!(v.dtype(), Dtype::F32, "imatrix tensor {n} not f32");
            let bytes = v.data();
            m.insert(n, bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        }
        println!("stq-bake: imatrix {} loaded ({} tensors)", p, m.len());
        m
    });
    if imatrix.is_none() {
        println!("stq-bake: WARNING no --imatrix — falling back to unweighted objective (diagnostic only)");
    }

    let kind_for = |class: &str| -> quant::StqKind {
        match (arm, class) {
            ("a", "gateup") => quant::StqKind::Stq1_0,
            ("a", "down") => quant::StqKind::Ls3Bit,
            ("a", "attn") => quant::StqKind::Ternary2,
            ("b", "gateup") => quant::StqKind::Ternary2,
            ("b", "down") => quant::StqKind::Ls3Bit,
            ("b", "attn") => quant::StqKind::Ls3Bit,
            // ISO-informed recipes (2026-08-30): ternary2 on gate/up costs +1.6% PPL; ls3bit on
            // attn is FATAL (GQA k/v integrity); ls3bit on down adds real damage. Arm c protects
            // down (NVFP4 untouched via --classes gateup,attn); arm d is the 1.31-bpw stretch.
            ("c", "gateup") => quant::StqKind::Ternary2,
            ("c", "attn") => quant::StqKind::Ternary2,
            ("d", "gateup") => quant::StqKind::Stq1_0,
            ("d", "attn") => quant::StqKind::Ternary2,
            // arm e: the 3-bit point of the gate/up bytes-vs-damage curve.
            ("e", "gateup") => quant::StqKind::Ls3Bit,
            ("e", "attn") => quant::StqKind::Ternary2,
            _ => unreachable!("arm {arm} has no {class} mapping"),
        }
    };
    // Targets: TRUNK language layers only — never mtp.* (draft), never model.visual.*, never GDN.
    // (Artifact names carry the full `model.language_model.layers.N.` prefix.)
    let classify = |n: &str| -> Option<&'static str> {
        if !n.ends_with(".weight_packed") || n.starts_with("mtp.")
            || !n.starts_with("model.language_model.layers.") {
            return None;
        }
        let class = if n.contains(".mlp.gate_proj.") || n.contains(".mlp.up_proj.") {
            "gateup"
        } else if n.contains(".mlp.down_proj.") {
            "down"
        } else if n.contains(".self_attn.") {
            "attn"
        } else {
            return None;
        };
        if classes.contains(&class) { Some(class) } else { None }
    };
    let is_target = |n: &str| -> bool { classify(n).is_some() };

    let recipe = format!(
        "stq-probe(arm-{arm}): gate/up={:?} down={:?} attn={:?} over nvfp4 — quality simulation, not a serving recipe",
        kind_for("gateup").name(),
        if matches!((arm, "down"), ("c", _) | ("d", _) | ("e", _)) { "nvfp4-untouched" } else { kind_for("down").name() },
        kind_for("attn").name());
    let meta = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("quant_recipe".to_string(), recipe.clone()),
    ]);
    println!("stq-bake: {} -> {} arm {arm} classes {:?} [{recipe}]", ind_s, out_s, classes);

    let index_path = ind.join("model.safetensors.index.json");
    let shards: Vec<std::path::PathBuf> = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).expect("read index");
        let idx: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
        idx["weight_map"].as_object().unwrap().values().filter_map(|v| v.as_str())
            .collect::<std::collections::BTreeSet<_>>().into_iter().map(|s| ind.join(s)).collect()
    } else { vec![ind.join("model.safetensors")] };

    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    #[derive(Default)]
    struct Stats { n: usize, sum: f64, max: f64 }
    let mut stats: std::collections::BTreeMap<&'static str, Stats> = Default::default();
    let (mut n_req, mut n_unweighted, mut n_skip_shards) = (0usize, 0usize, 0usize);
    let t0 = std::time::Instant::now();

    for (si, sf) in shards.iter().enumerate() {
        let shard_no = si + 1;
        if shard_no < shard_start || shard_no > shard_end { continue; }
        let fname = sf.file_name().unwrap_or_default().to_string_lossy().to_string();
        if outd.join(&fname).exists() {
            println!("  shard {shard_no}/{}: {fname} present, skipping", shards.len());
            n_skip_shards += 1;
            continue;
        }
        let t_shard = std::time::Instant::now();
        println!("  shard {shard_no}/{}: {fname}", shards.len());
        let raw = std::fs::read(sf).expect("read shard");
        let st = SafeTensors::deserialize(&raw).expect("parse shard");
        let tvec: Vec<(String, TensorView)> = st.tensors();
        let mut outs: Vec<Out> = Vec::with_capacity(tvec.len());
        for (n, v) in &tvec {
            let stem = &n[..n.rfind(".weight").unwrap_or(n.len())];
            if let Some(class) = classify(n) {
                if n_req >= limit { // --limit smoke mode: copy the rest verbatim
                    outs.push(Out { name: n.clone(), dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() });
                    continue;
                }
                let get = |suffix: &str| -> &TensorView {
                    let full = format!("{stem}.weight{suffix}");
                    &tvec.iter().find(|(tn, _)| *tn == full)
                        .unwrap_or_else(|| panic!("{full} missing in {fname} (triple split across shards?)")).1
                };
                let (pv, sv, gv) = (get("_packed"), get("_scale"), get("_global_scale"));
                let (m, kh) = (pv.shape()[0], pv.shape()[1]);
                let k = kh * 2;
                let q = quant::Nvfp4Tensor {
                    qweight: pv.data().to_vec(),
                    scales: sv.data().to_vec(),
                    global_scale: f32::from_le_bytes(gv.data()[..4].try_into().unwrap()),
                    m, k,
                };
                let orig = quant::dequantize_nvfp4_f32(&q);
                let imat_key = format!("{stem}.weight");
                let qw: Option<&[f32]> = match &imatrix {
                    Some(map) => match map.get(&imat_key) {
                        Some(vq) if vq.len() == k => Some(vq),
                        Some(vq) => panic!("imatrix {imat_key}: len {} != K {k}", vq.len()),
                        None => { n_unweighted += 1; None }
                    },
                    None => None,
                };
                let mut w = orig.clone();
                quant::fake_quant_stq(&mut w, m, k, qw, kind_for(class));
                let ssd: f64 = orig.iter().zip(w.iter()).map(|(&a, &b)| { let e = a - b; (e * e) as f64 }).sum();
                let x2: f64 = orig.iter().map(|&a| (a * a) as f64).sum();
                let rel_l2 = (ssd / x2.max(f64::MIN_POSITIVE)).sqrt();
                let stq_stat = stats.entry(class).or_default();
                stq_stat.n += 1; stq_stat.sum += rel_l2; stq_stat.max = stq_stat.max.max(rel_l2);
                let wbf: Vec<half::bf16> = w.iter().map(|&x| half::bf16::from_f32(x)).collect();
                let q2 = quant::quantize_nvfp4(&wbf, m, k);
                outs.push(Out { name: n.clone(), dtype: Dtype::U8, shape: vec![m, kh], data: q2.qweight });
                outs.push(Out { name: format!("{stem}.weight_scale"), dtype: Dtype::F8_E4M3,
                                shape: vec![m, k / quant::BLOCK], data: q2.scales });
                outs.push(Out { name: format!("{stem}.weight_global_scale"), dtype: Dtype::F32,
                                shape: vec![1], data: q2.global_scale.to_le_bytes().to_vec() });
                n_req += 1;
                println!("    {} {} [{m}x{k}] rel-L2 {rel_l2:.4}", kind_for(class).name(), stem, );
            } else if n.ends_with("_scale") {
                // Scale siblings ride out with their packed triple — never copy them alone.
                let packed_name = format!("{stem}.weight_packed");
                if is_target(&packed_name) && tvec.iter().any(|(tn, _)| *tn == packed_name) {
                    continue;
                }
                outs.push(Out { name: n.clone(), dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() });
            } else {
                outs.push(Out { name: n.clone(), dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() });
            }
        }
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join(&fname)).expect("write shard");
        let hwm = std::fs::read_to_string("/proc/self/status").ok()
            .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).map(|l| l.trim().to_string()))
            .unwrap_or_else(|| "VmHWM n/a".into());
        println!("    wrote {fname} ({} tensors, {:.1}s, {hwm})", outs.len(), t_shard.elapsed().as_secs_f32());
    }
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json",
              "model.safetensors.index.json"] {
        let src = ind.join(f);
        if src.exists() && !outd.join(f).exists() { let _ = std::fs::copy(&src, outd.join(f)); }
    }
    println!("stq-bake per-class rel-L2 (STQ round-trip vs dequantized NVFP4, pre-requant):");
    for (class, st) in &stats {
        println!("  {class:8} n={:4} mean={:.4} max={:.4}", st.n, st.sum / st.n as f64, st.max);
    }
    println!("stq-bake done: {n_req} tensors, {n_unweighted} without imatrix, {n_skip_shards} shards skipped, {:.1}s",
             t0.elapsed().as_secs_f32());
}

/// `--stq-bake --check --model-dir <dir>` — the fitter unit gate, no bake: on real gate_proj rows,
/// the LS+imatrix encoder must beat the reference (amax + argmin|x|) encoder by a wide margin
/// (AngelSlim measured −89.7% weighted SSD from the LS scale alone — expect ≳5× here).
fn stq_check_mode(args: &[String]) {
    use safetensors::{SafeTensors, tensor::TensorView};
    use gb10_inference::quant;
    let ind_s = parse_arg(args, "--model-dir").expect("--check requires --model-dir <packed-dir>");
    let ind = std::path::Path::new(ind_s);
    let index_path = ind.join("model.safetensors.index.json");
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path).expect("read index")).expect("parse index");
    // sample across the depth of the network: layers {0, 16, 32, 48, 63} gate_proj
    let mut picks: Vec<(String, std::path::PathBuf)> = Vec::new();
    for layer in [0usize, 16, 32, 48, 63] {
        let name = format!("model.language_model.layers.{layer}.mlp.gate_proj.weight_packed");
        if let Some(f) = idx["weight_map"][&name].as_str() { picks.push((name, ind.join(f))); }
    }
    assert!(!picks.is_empty(), "no gate_proj tensors found in index");
    let rows_per_tensor = 4usize;
    let (mut ssd_ref, mut ssd_ls, mut nblk) = (0.0f64, 0.0f64, 0usize);
    for (name, path) in &picks {
        let raw = std::fs::read(path).expect("read shard");
        let st = SafeTensors::deserialize(&raw).expect("parse shard");
        let stem = name.trim_end_matches("_packed");
        let get = |suffix: &str| -> TensorView {
            let full = format!("{stem}{suffix}");
            st.tensor(&full).unwrap_or_else(|_| panic!("{full} missing"))
        };
        let (pv, sv, gv) = (get("_packed"), get("_scale"), get("_global_scale"));
        let (m, kh) = (pv.shape()[0], pv.shape()[1]);
        let k = kh * 2;
        let q = quant::Nvfp4Tensor {
            qweight: pv.data().to_vec(), scales: sv.data().to_vec(),
            global_scale: f32::from_le_bytes(gv.data()[..4].try_into().unwrap()), m, k,
        };
        let w = quant::dequantize_nvfp4_f32(&q);
        let stride = (m / rows_per_tensor).max(1);
        let mut y_ref = vec![0.0f32; quant::STQ_BLOCK];
        let mut y_ls = vec![0.0f32; quant::STQ_BLOCK];
        for r in (0..m).step_by(stride) {
            for b in 0..k / quant::STQ_BLOCK {
                let x = &w[r * k + b * quant::STQ_BLOCK..][..quant::STQ_BLOCK];
                quant::stq1_0_block_reference(x, &mut y_ref);
                quant::stq1_0_block(x, None, &mut y_ls);
                ssd_ref += quant::stq_weighted_ssd(x, &y_ref, None);
                ssd_ls += quant::stq_weighted_ssd(x, &y_ls, None);
                nblk += 1;
            }
        }
        println!("  {stem} [{m}x{k}] sampled {rows_per_tensor} rows");
    }
    let ratio = ssd_ref / ssd_ls.max(f64::MIN_POSITIVE);
    println!("check: {nblk} blocks, weighted SSD ref(amax)={ssd_ref:.4e} ls={ssd_ls:.4e} improvement x{ratio:.1}");
    if ratio >= 5.0 {
        println!("check: PASS (LS encoder dominates the reference, as the AngelSlim claim implies)");
    } else {
        println!("check: FAIL — LS encoder not dominating; do not bake until this passes");
        std::process::exit(1);
    }
}

/// `--stq-bake --verify --model-dir <a> --model-dir-b <b>` — post-hoc fidelity check: dequantize
/// every target triple in both artifacts and report rel-L2 of B vs A per class. Catches requant
/// mis-encoding that pre-requant stats cannot see (the iso-b-attn explosion postmortem tool).
fn stq_verify_mode(args: &[String]) {
    use safetensors::{SafeTensors, tensor::TensorView};
    use gb10_inference::quant;
    let dir_a = std::path::Path::new(parse_arg(args, "--model-dir").expect("--verify requires --model-dir <a>"));
    let dir_b = std::path::Path::new(parse_arg(args, "--model-dir-b").expect("--verify requires --model-dir-b <b>"));
    let read_triples = |dir: &std::path::Path| -> Vec<(String, quant::Nvfp4Tensor)> {
        let idx: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("index")).expect("parse index");
        let mut out = Vec::new();
        for fname in idx["weight_map"].as_object().unwrap().values()
            .filter_map(|v| v.as_str()).collect::<std::collections::BTreeSet<_>>() {
            let raw = std::fs::read(dir.join(fname)).expect("read shard");
            let st = SafeTensors::deserialize(&raw).expect("parse shard");
            for (n, v) in st.tensors() {
                if !n.ends_with(".weight_packed") { continue; }
                let stem = &n[..n.rfind(".weight").unwrap()];
                let get = |suffix: &str| -> Option<TensorView> {
                    st.tensor(&format!("{stem}.weight{suffix}")).ok()
                };
                let (Some(pv), Some(sv), Some(gv)) = (get("_packed"), get("_scale"), get("_global_scale")) else { continue };
                let (m, kh) = (pv.shape()[0], pv.shape()[1]);
                out.push((stem.to_string(), quant::Nvfp4Tensor {
                    qweight: pv.data().to_vec(), scales: sv.data().to_vec(),
                    global_scale: f32::from_le_bytes(gv.data()[..4].try_into().unwrap()),
                    m, k: kh * 2,
                }));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    };
    println!("verify: reading A = {}", dir_a.display());
    let ta = read_triples(dir_a);
    println!("verify: reading B = {}", dir_b.display());
    let tb = read_triples(dir_b);
    assert_eq!(ta.len(), tb.len(), "triple count mismatch");
    let mut worst = Vec::new();
    for ((na, qa), (nb, qb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb, "tensor name mismatch {na} vs {nb}");
        assert_eq!((qa.m, qa.k), (qb.m, qb.k), "{na}: shape mismatch");
        let va = quant::dequantize_nvfp4_f32(qa);
        let vb = quant::dequantize_nvfp4_f32(qb);
        let ssd: f64 = va.iter().zip(vb.iter()).map(|(&x, &y)| { let e = x - y; (e * e) as f64 }).sum();
        let x2: f64 = va.iter().map(|&x| (x * x) as f64).sum();
        let rel = (ssd / x2.max(f64::MIN_POSITIVE)).sqrt();
        let class = if na.contains("mlp.gate_proj") || na.contains("mlp.up_proj") { "gateup" }
            else if na.contains("mlp.down_proj") { "down" }
            else if na.contains("self_attn.q_proj") { "attn.q" }
            else if na.contains("self_attn.k_proj") { "attn.k" }
            else if na.contains("self_attn.v_proj") { "attn.v" }
            else if na.contains("self_attn.o_proj") { "attn.o" }
            else { "other" };
        worst.push((rel, na.clone(), class));
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("verify: {} triples; top-15 divergent (B vs A rel-L2):", worst.len());
    for (rel, n, c) in worst.iter().take(15) {
        println!("  {rel:.4}  [{c:7}] {n}");
    }
    for class in ["gateup", "down", "attn.q", "attn.k", "attn.v", "attn.o"] {
        let sel: Vec<f64> = worst.iter().filter(|(_, _, c)| *c == class).map(|(r, _, _)| *r).collect();
        if !sel.is_empty() {
            let mean = sel.iter().sum::<f64>() / sel.len() as f64;
            let mx = sel.iter().cloned().fold(0.0f64, f64::max);
            println!("  {class:7}: mean {mean:.4}  max {mx:.4}  n={}", sel.len());
        }
    }
}
