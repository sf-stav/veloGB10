//! Offline quantizer entry points: `run_quantize` (NVFP4/FP8 recipes), GDN requantization and simulation,
//! and the DF2 bake / quantization-fidelity tools. The STQ bake and verify modes are in `stq.rs`.

use super::*;

pub(crate) fn run_quantize(args: &[String]) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    use gb10_inference::quant::{self, Fmt};

    let in_dir = parse_arg(args, "--model-dir").expect("--model-dir <in> required");
    let out_dir = parse_arg(args, "--out").expect("--out <dir> required");
    let recipe_s = parse_arg(args, "--recipe").unwrap_or("all,gdn:fp8");
    let recipe = quant::parse_recipe(recipe_s).expect("empty recipe");

    let ind = std::path::Path::new(in_dir);
    let outd = std::path::Path::new(out_dir);
    std::fs::create_dir_all(outd).expect("create --out dir");

    println!("Quantizing {} -> {}", in_dir, out_dir);
    println!("  recipe: {}", recipe_s);
    for (g, f) in &recipe {
        println!("    {:<8} -> {}", quant::group_name(*g), quant::fmt_name(*f));
    }

    // Shards, in index order when there is an index.
    let index_path = ind.join("model.safetensors.index.json");
    let shards: Vec<std::path::PathBuf> = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).expect("read index");
        let idx: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
        idx["weight_map"].as_object().unwrap().values()
            .filter_map(|v| v.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter().map(|s| ind.join(s)).collect()
    } else {
        std::fs::read_dir(ind).unwrap().filter_map(|e| {
            let e = e.ok()?;
            let n = e.file_name().to_string_lossy().to_string();
            if n.ends_with(".safetensors") { Some(e.path()) } else { None }
        }).collect()
    };

    // Owned output buffers (TensorView borrows, so these must outlive serialization).
    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    let mut outs: Vec<Out> = Vec::new();
    let (mut n_q4, mut n_q8, mut n_copy) = (0usize, 0usize, 0usize);
    let (mut bytes_in, mut bytes_out) = (0u64, 0u64);
    let t0 = std::time::Instant::now();

    // Quantize-or-copy ONE tensor into `outs`, honoring the recipe. Used both inline in the shard loop
    // and for the SYNTHESIZED fused MTP-expert stacks below, so both paths are byte-identical.
    // 3D experts [E,M,K] flatten to [E*M,K] (the per-16-block E4M3 scales absorb per-expert magnitude
    // variation; M is 16-aligned so a block never straddles two experts). Vision tower stays bf16.
    //
    // S-H0.4a: the qwen4_exp PLE n-gram table is handled INSIDE emit, before the generic gate —
    // its shards are [2,500,012, 160] and 2,500,012 % 16 != 0, so the GEMM `m_eff % 16` gate would
    // silently copy the table through as bf16 (95 GiB of dead weight) and the artifact would then
    // FAIL TO LOAD: attach_ple_table (gpu.rs) requires the ple_ngram_nvfp4.json sidecar. The row
    // codec quantizes each row independently at a fixed 96 B (80 B E2M1 + 10 B E4M3 + 6 B pad), so
    // no M/K alignment is needed. Records stream to ple_ngram_nvfp4.bin; the tensors do NOT enter
    // the artifact safetensors; the sidecar is written in FINALIZE below.
    // The PLE row-record state lives in a RefCell so `emit` stays a shared (Fn) closure —
    // fuse_base below calls emit and must not become FnMut itself.
    let ple_state = std::cell::RefCell::new((
        std::fs::File::create(outd.join("ple_ngram_nvfp4.bin")).expect("create ple_ngram_nvfp4.bin"),
        Vec::<(u64, usize, f32, String)>::new(), // (shard i, rows, global scale, tensor prefix)
        0usize,                                  // total rows
    ));
    let emit = |name: String, dtype: Dtype, shape: Vec<usize>, data: &[u8],
                outs: &mut Vec<Out>, n_q4: &mut usize, n_q8: &mut usize, n_copy: &mut usize,
                bytes_out: &mut u64| {
        let fmt = quant::fmt_for(&recipe, &name);
        let last = *shape.last().unwrap_or(&0);
        let m_eff = if shape.len() == 3 { shape[0] * shape[1] }
                    else if shape.len() == 2 { shape[0] } else { 0 };
        if fmt != Fmt::Bf16 && dtype == Dtype::BF16 && shape.len() == 2
                && last == quant::PLE_DIM && name.contains(".ngram_embedding.shard_") {
            let ple_shard_i: u64 = name.rsplit_once("shard_").unwrap().1
                .split('.').next().unwrap().parse().expect("ple shard index");
            let (rec, gs) = quant::quantize_ple_rows(&gb10_inference::podread::pod_cow::<half::bf16>(data), m_eff);
            use std::io::Write;
            let mut st = ple_state.borrow_mut();
            st.0.write_all(&rec).expect("write ple records");
            let prefix = name.rsplit_once(".shard_").unwrap().0.to_string();
            st.1.push((ple_shard_i, m_eff, gs, prefix));
            st.2 += m_eff;
            *bytes_out += rec.len() as u64;
            return;
        }
        let quantizable = fmt != Fmt::Bf16
            && (shape.len() == 2 || shape.len() == 3)
            && dtype == Dtype::BF16
            && last % quant::BLOCK == 0
            && m_eff % 16 == 0
            && !name.contains(".visual.");
        if !quantizable {
            // S-B4 footgun guard: a PLE-eligible tensor that MISSED the row-codec branch above
            // would previously copy through silently as bf16 — bake #1 shipped a 95 GiB dead
            // table that way and produced an artifact `attach_ple_table` rejects. If the name
            // says ngram_embedding but the row-codec gate did not take it, scream: the only
            // ways to land here are shape/naming mismatches, and every one of them yields an
            // unloadable artifact. (A bf16-recipe copy-through is legitimate and stays quiet.)
            if fmt != Fmt::Bf16 && dtype == Dtype::BF16 && name.contains("ngram_embedding") {
                eprintln!(
                    "[quantize] WARNING: PLE-eligible tensor `{name}` shape {shape:?} was NOT \
                     handled by the PLE row codec (expected 2-D bf16 [rows, {}] named \
                     `*.ngram_embedding.shard_<i>.weight`); it is being copied through \
                     UNQUANTIZED into the safetensors and `attach_ple_table` will REJECT the \
                     finished artifact (missing ple_ngram_nvfp4.json sidecar). Fix the gate \
                     condition or the checkpoint naming before shipping this bake.",
                    quant::PLE_DIM
                );
            }
            // Copy through unchanged: norms, conv1d, A_log, dt_bias, router (if bf16), vision, bf16 head.
            outs.push(Out { name, dtype, shape, data: data.to_vec() });
            *n_copy += 1;
            *bytes_out += data.len() as u64;
            return;
        }
        let (m, k) = (m_eff, last);
        let w_cow = gb10_inference::podread::pod_cow::<half::bf16>(data); // S-B9-REL-PPL: offsets may be unaligned
        let w: &[half::bf16] = &w_cow;
        let stem = name.strip_suffix(".weight").unwrap_or(&name).to_string();
        match fmt {
            Fmt::Nvfp4 => {
                let q = quant::quantize_nvfp4(w, m, k);
                *bytes_out += (q.qweight.len() + q.scales.len() + 4) as u64;
                outs.push(Out { name: format!("{}.weight_packed", stem), dtype: Dtype::U8,
                                shape: vec![m, k / 2], data: q.qweight });
                outs.push(Out { name: format!("{}.weight_scale", stem), dtype: Dtype::F8_E4M3,
                                shape: vec![m, k / quant::BLOCK], data: q.scales });
                outs.push(Out { name: format!("{}.weight_global_scale", stem), dtype: Dtype::F32,
                                shape: vec![1], data: q.global_scale.to_le_bytes().to_vec() });
                *n_q4 += 1;
            }
            Fmt::Fp8 => {
                let q = quant::quantize_fp8(w, m, k);
                let sc: Vec<u8> = q.row_scale.iter().flat_map(|f| f.to_le_bytes()).collect();
                *bytes_out += (q.qweight.len() + sc.len()) as u64;
                outs.push(Out { name: format!("{}.weight", stem), dtype: Dtype::F8_E4M3,
                                shape: vec![m, k], data: q.qweight });
                outs.push(Out { name: format!("{}.weight_scale", stem), dtype: Dtype::F32,
                                shape: vec![m], data: sc });
                *n_q8 += 1;
            }
            Fmt::Bf16 => unreachable!(),
        }
    };

    // Some checkpoints store routed experts UN-FUSED and per-expert (`...mlp.experts.<i>.{gate,up,down}_proj.weight`):
    // the 122B's MTP head, and Hy3 (hy_v3) for EVERY MoE layer (79 + the layer-80 MTP block). Our
    // loader/kernels only ingest the fused layout (`...mlp.experts.gate_up_proj` / `.down_proj`), so FUSE
    // per-expert tensors before quantizing: stash keyed by (base, proj) with a BTreeMap over the INTEGER
    // expert index (a lexical sort would permute 0,1,10,100,… → silent wrong experts). Fusion happens
    // ON-COMPLETE (gate/up/down each holding a contiguous 0..E set) so a 295B model's stash never exceeds
    // ~1 layer of experts in RAM; anything left unfused at the end is a loud error, not silent garbage.
    type ExpertMap = std::collections::BTreeMap<usize, (Vec<usize>, Vec<u8>)>;
    type Pending = std::collections::BTreeMap<String, std::collections::BTreeMap<String, ExpertMap>>;
    let mut pending: Pending = std::collections::BTreeMap::new();
    let n_experts: Option<usize> = std::fs::read_to_string(ind.join("config.json")).ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|c| {
            // Multimodal-wrapped configs (e.g. KAT-Coder) nest the model section under `text_config`.
            let get = |m: &serde_json::Value| m.get("num_experts").or_else(|| m.get("n_routed_experts")).and_then(|v| v.as_u64());
            get(&c).or_else(|| c.get("text_config").and_then(get))
        })
        .map(|v| v as usize);
    let parse_expert = |name: &str| -> Option<(String, String, usize)> {
        let stem = name.strip_suffix(".weight")?;
        let (base_prefix, tail) = stem.split_once(".experts.")?;   // "...layers.L.mlp" , "<i>.<proj>"
        let (idx_s, proj) = tail.split_once('.')?;
        if proj != "gate_proj" && proj != "up_proj" && proj != "down_proj" { return None; }
        let idx: usize = idx_s.parse().ok()?;
        Some((format!("{}.experts", base_prefix), proj.to_string(), idx))
    };
    let complete = |projs: &std::collections::BTreeMap<String, ExpertMap>, e: usize| -> bool {
        ["gate_proj", "up_proj", "down_proj"].iter().all(|p| projs.get(*p)
            .map(|m| m.len() == e && (0..e).all(|i| m.contains_key(&i))).unwrap_or(false))
    };
    // Fuse one base's per-expert tensors into the stacked layout the loader expects, then run it through
    // the SAME `emit` (quantize or copy-through per recipe). Ordering is load-bearing in BOTH dims:
    // gate_up = concat([gate, up], dim=0); experts stacked in INTEGER order.
    let fuse_base = |base: &String, projs: &std::collections::BTreeMap<String, ExpertMap>,
                     outs: &mut Vec<Out>, n_q4: &mut usize, n_q8: &mut usize, n_copy: &mut usize,
                     bytes_out: &mut u64| {
        let gate = projs.get("gate_proj").unwrap_or_else(|| panic!("{base}: no gate_proj experts"));
        let up   = projs.get("up_proj").unwrap_or_else(|| panic!("{base}: no up_proj experts"));
        let down = projs.get("down_proj").unwrap_or_else(|| panic!("{base}: no down_proj experts"));
        let e = gate.len();
        // Every projection must have the SAME contiguous 0..E expert set (assert one per expert).
        for (nm, m) in [("gate", gate), ("up", up), ("down", down)] {
            assert_eq!(m.len(), e, "{base}: {nm} has {} experts, gate has {e}", m.len());
            for i in 0..e { assert!(m.contains_key(&i), "{base}: {nm} missing expert {i}"); }
        }
        let (inter, hidden) = { let s = &gate[&0].0; assert_eq!(s.len(), 2, "gate not 2-D"); (s[0], s[1]) };
        // Shape guard — catches a transposed/permuted export before it becomes silent garbage.
        for i in 0..e {
            assert_eq!(gate[&i].0, vec![inter, hidden], "{base}: gate[{i}] shape");
            assert_eq!(up[&i].0,   vec![inter, hidden], "{base}: up[{i}] shape");
            assert_eq!(down[&i].0, vec![hidden, inter], "{base}: down[{i}] shape");
        }
        // gate_up[e] = concat([gate[e], up[e]], dim=0) → [2·inter, hidden]; stack over e → [E,2·inter,hidden].
        let mut gu = Vec::<u8>::with_capacity(e * 2 * inter * hidden * 2);
        for i in 0..e { gu.extend_from_slice(&gate[&i].1); gu.extend_from_slice(&up[&i].1); }
        emit(format!("{base}.gate_up_proj"), Dtype::BF16, vec![e, 2 * inter, hidden], &gu,
             outs, n_q4, n_q8, n_copy, bytes_out);
        let mut dn = Vec::<u8>::with_capacity(e * hidden * inter * 2);
        for i in 0..e { dn.extend_from_slice(&down[&i].1); }
        emit(format!("{base}.down_proj"), Dtype::BF16, vec![e, hidden, inter], &dn,
             outs, n_q4, n_q8, n_copy, bytes_out);
        println!("  fused experts: {base} → gate_up_proj [{e},{},{hidden}] + down_proj [{e},{hidden},{inter}]", 2 * inter);
    };

    // STREAM the output to disk in ~4 GiB shards as we go, so peak RAM is ~one shard + one input shard.
    // A 397B's ~220 GB of quantized output can't be buffered in 128 GB host RAM (the old accumulate-then-
    // shard path OOM'd past ~100 GB). Small models still collapse to one `model.safetensors` at the end.
    // S-H0.4a: 12 GiB shards made the biggest output shard 13.6 GB, and the loader's single-node
    // transient model (3x max shard + pipeline) then exceeded what a 128 GB box has free even with
    // the PLE offloaded — the load refused before it started. 4 GiB shards keep max-shard ~6 GB and
    // the transient ~28 GB, so qwen4_exp TP=1 loads pass the guard honestly (no --load-force).
    const SHARD_BYTES: usize = 4 * 1024 * 1024 * 1024;
    let meta = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("quant_recipe".to_string(), recipe_s.to_string()),
    ]);
    let mut weight_map = serde_json::Map::new();
    let mut shard_idx = 0usize;
    let outs_bytes = |outs: &Vec<Out>| -> usize { outs.iter().map(|o| o.data.len()).sum() };
    let write_shard = |outs: &mut Vec<Out>,
                       weight_map: &mut serde_json::Map<String, serde_json::Value>,
                       shard_idx: &mut usize| {
        if outs.is_empty() { return; }
        let fname = format!("model-{:05}.safetensors", *shard_idx + 1);
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view")))
            .collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join(&fname)).expect("write shard");
        for o in outs.iter() { weight_map.insert(o.name.clone(), serde_json::json!(fname.clone())); }
        *shard_idx += 1;
        println!("  wrote {} ({} tensors)", fname, outs.len());
        outs.clear();
    };

    // BASICS (owner 2026-11-02): the input tensors' bytes as a percentage at the normal level
    let qprog = gb10_inference::boot_report::Progress::new("quantizing",
        shards.iter().map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum(),
        gb10_inference::boot_report::Unit::Bytes);
    for (i, sf) in shards.iter().enumerate() {
        println!("  shard {}/{}: {}", i + 1, shards.len(),
                 sf.file_name().unwrap_or_default().to_string_lossy());
        let raw = std::fs::read(sf).expect("read shard");
        let st = SafeTensors::deserialize(&raw).expect("parse shard");
        for (name, view) in st.tensors() {
            let data = view.data();
            bytes_in += data.len() as u64;
            qprog.add(data.len() as u64);
            // Un-fused expert weight? stash for fusion, don't emit yet. Fuse ON-COMPLETE (a full
            // contiguous 0..E set for all three projections) so the stash stays ~1 layer deep in RAM.
            if let Some((base, proj, idx)) = parse_expert(&name) {
                pending.entry(base.clone()).or_default().entry(proj).or_default()
                    .insert(idx, (view.shape().to_vec(), data.to_vec()));
                if let Some(e) = n_experts {
                    if complete(pending.get(&base).unwrap(), e) {
                        let projs = pending.remove(&base).unwrap();
                        fuse_base(&base, &projs, &mut outs, &mut n_q4, &mut n_q8, &mut n_copy, &mut bytes_out);
                        if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
                    }
                }
                continue;
            }
            emit(name.clone(), view.dtype(), view.shape().to_vec(), data,
                 &mut outs, &mut n_q4, &mut n_q8, &mut n_copy, &mut bytes_out);
            if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
        }
        // shard bytes dropped here — peak memory is one shard + the (much smaller) output
    }
    qprog.finish();

    // Drain any bases still pending at the end (e.g. the 122B's single MTP block, whose fusion triggers
    // here when `n_experts` was absent or the set only completed on the last shard). They MUST be
    // complete — an incomplete set means a broken checkpoint (missing expert), and emitting anything
    // would produce a wrong-expert model. Loud, not silent.
    for (base, projs) in pending.iter() {
        let e = n_experts.unwrap_or_else(|| projs["gate_proj"].len());
        assert!(complete(projs, e),
                "{base}: incomplete expert set ({e} expected) — cannot fuse; the checkpoint is broken");
        fuse_base(base, projs, &mut outs, &mut n_q4, &mut n_q8, &mut n_copy, &mut bytes_out);
        if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
    }

    // FINALIZE. If nothing was flushed mid-run (small model < SHARD_BYTES), collapse to one
    // `model.safetensors` (no index) — preserves the existing small-model layout. Otherwise flush the
    // tail shard and write the index over all streamed shards.
    if shard_idx == 0 {
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join("model.safetensors")).expect("write safetensors");
    } else {
        write_shard(&mut outs, &mut weight_map, &mut shard_idx);
        let index = serde_json::json!({ "metadata": { "total_size": bytes_out }, "weight_map": weight_map });
        std::fs::write(outd.join("model.safetensors.index.json"),
                       serde_json::to_string_pretty(&index).unwrap()).expect("write index");
        println!("  wrote {} shards + index (streamed to disk; peak RAM ~one shard)", shard_idx);
    }

    // Carry the sidecars across, and record the recipe in config.json so the loader can self-detect.
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json"] {
        let src = ind.join(f);
        if src.exists() { let _ = std::fs::copy(&src, outd.join(f)); }
    }
    // S-H0.4a: the PLE row-record table's sidecar — the loader's PleTableMeta::load and
    // ref_forward.py's dequant_ple_table both key off this file. Shard scales are stored in
    // NUMERIC shard order (the hash offsets are running sums over that order).
    let (ple_bin, mut ple_shards, ple_rows_total) = ple_state.into_inner();
    drop(ple_bin);
    if !ple_shards.is_empty() {
        ple_shards.sort_by_key(|s| s.0);
        let rows_per_shard = ple_shards[0].1;
        for s in &ple_shards[1..] {
            assert_eq!(s.1, rows_per_shard, "PLE shards must be uniform");
            assert_eq!(s.3, ple_shards[0].3, "PLE shard prefixes must agree");
        }
        let side = serde_json::json!({
            "format": "ple-rows-nvfp4-v1",
            "complete": true,
            "dim": quant::PLE_DIM,
            "record_bytes": quant::PLE_REC_BYTES,
            "num_shards": ple_shards.len(),
            "rows_per_shard": rows_per_shard,
            "total_rows": ple_rows_total,
            "file": "ple_ngram_nvfp4.bin",
            "tensor_prefix": ple_shards[0].3,
            "shard_global_scales": ple_shards.iter().map(|s| s.2).collect::<Vec<f32>>(),
        });
        std::fs::write(outd.join("ple_ngram_nvfp4.json"),
                       serde_json::to_string_pretty(&side).unwrap()).expect("write ple sidecar");
        println!("  ple table: {} shards x {} rows -> ple_ngram_nvfp4.bin ({:.2} GB), sidecar written",
                 ple_shards.len(), rows_per_shard,
                 (ple_rows_total * quant::PLE_REC_BYTES) as f64 / 1e9);
    }
    let cfg_path = outd.join("config.json");
    if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
        if let Ok(mut cfg) = serde_json::from_str::<serde_json::Value>(&raw) {
            cfg["quantization_config"] = serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "recipe": recipe_s,
            });
            let _ = std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap());
        }
    }

    let gi = bytes_in as f64 / 1e9;
    let go = bytes_out as f64 / 1e9;
    println!();
    println!("  tensors: {} NVFP4, {} FP8, {} copied through", n_q4, n_q8, n_copy);
    println!("  size:    {:.2} GB -> {:.2} GB   ({:.2}x smaller)", gi, go, gi / go.max(1e-9));
    println!("  wrote {} in {:.0}s", outd.display(), t0.elapsed().as_secs_f32());
}

/// `--perplexity --model-dir <d> --text <file>` — perplexity on held-out text.
///
/// The quality gate for quantization. Combine with `--fake-quant=<groups>` to measure what
/// 4 bits actually costs, per tensor group, in the real engine — without needing any 4-bit kernel.
// Derive a gdn4 (GDN-nvfp4) artifact from a mixed (GDN-fp8) one — no bf16 needed. Only the fp8 GDN
// in/out-proj tensors are re-quantized (dequant fp8 → bf16 → nvfp4); everything else (already nvfp4 or
// bf16) is copied byte-for-byte. Output is streamed in ~12 GB shards (bounded RAM, works for the 397B).
pub(crate) fn run_requant_gdn(args: &[String]) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    use gb10_inference::quant;
    let from = parse_arg(args, "--from").expect("--requant-gdn requires --from <mixed-dir>");
    let out  = parse_arg(args, "--out").expect("--requant-gdn requires --out <gdn4-dir>");
    let ind = std::path::Path::new(from);
    let outd = std::path::Path::new(out);
    std::fs::create_dir_all(outd).expect("create --out dir");
    println!("Deriving gdn4 (GDN nvfp4) from {} -> {}", from, out);

    let index_path = ind.join("model.safetensors.index.json");
    let shards: Vec<std::path::PathBuf> = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).expect("read index");
        let idx: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
        idx["weight_map"].as_object().unwrap().values().filter_map(|v| v.as_str())
            .collect::<std::collections::BTreeSet<_>>().into_iter().map(|s| ind.join(s)).collect()
    } else { vec![ind.join("model.safetensors")] };

    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    const SHARD_BYTES: usize = 12 * 1024 * 1024 * 1024;
    let meta = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("quant_recipe".to_string(), "all,-router (gdn4 derived from mixed via requant-gdn)".to_string()),
    ]);
    let mut weight_map = serde_json::Map::new();
    let mut shard_idx = 0usize;
    let mut bytes_out = 0u64;
    let outs_bytes = |o: &Vec<Out>| -> usize { o.iter().map(|x| x.data.len()).sum() };
    let write_shard = |outs: &mut Vec<Out>, wm: &mut serde_json::Map<String, serde_json::Value>, si: &mut usize| {
        if outs.is_empty() { return; }
        let fname = format!("model-{:05}.safetensors", *si + 1);
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join(&fname)).expect("write shard");
        for o in outs.iter() { wm.insert(o.name.clone(), serde_json::json!(fname.clone())); }
        *si += 1; println!("  wrote {} ({} tensors)", fname, outs.len()); outs.clear();
    };
    let is_gdn = |n: &str| n.contains(".linear_attn.") && (n.contains("in_proj") || n.contains("out_proj"));

    let mut outs: Vec<Out> = Vec::new();
    let (mut n_req, mut n_copy) = (0usize, 0usize);
    for (si, sf) in shards.iter().enumerate() {
        println!("  shard {}/{}: {}", si + 1, shards.len(), sf.file_name().unwrap_or_default().to_string_lossy());
        let raw = std::fs::read(sf).expect("read shard");
        let st = SafeTensors::deserialize(&raw).expect("parse shard");
        let tvec: Vec<(String, TensorView)> = st.tensors();
        // Group the shard's tensors by STEM so a tensor's whole family (nvfp4 triple
        // weight_packed/weight_scale/weight_global_scale, or the fp8 weight+weight_scale) is emitted
        // together and never split across output shards — the loader pairs the triple WITHIN one shard.
        let mut by_stem: std::collections::BTreeMap<&str, Vec<&(String, TensorView)>> =
            std::collections::BTreeMap::new();
        for t in &tvec {
            let n = t.0.as_str();
            let stem = &n[..n.rfind(".weight").unwrap_or(n.len())];
            by_stem.entry(stem).or_default().push(t);
        }
        for (stem, parts) in by_stem {
            // A GDN in/out-proj stored as fp8 = a `.weight` (F8) + `.weight_scale` (F32). Re-quantize it.
            let fp8_w = parts.iter().find(|(n, v)| n.ends_with(".weight") && v.dtype() == Dtype::F8_E4M3);
            if is_gdn(stem) && fp8_w.is_some() {
                let (_, wv) = fp8_w.unwrap();
                let (_, sv) = parts.iter().find(|(n, _)| n.ends_with(".weight_scale"))
                    .expect("gdn fp8 weight_scale missing");
                let (m, k) = (wv.shape()[0], wv.shape()[1]);
                let fp8 = quant::Fp8Tensor {
                    qweight: wv.data().to_vec(),
                    row_scale: gb10_inference::podread::pod_vec::<f32>(sv.data()),
                    m, k,
                };
                let bf = quant::dequantize_fp8(&fp8);
                let q = quant::quantize_nvfp4(&bf, m, k);
                bytes_out += (q.qweight.len() + q.scales.len() + 4) as u64;
                outs.push(Out { name: format!("{stem}.weight_packed"), dtype: Dtype::U8, shape: vec![m, k/2], data: q.qweight });
                outs.push(Out { name: format!("{stem}.weight_scale"), dtype: Dtype::F8_E4M3, shape: vec![m, k/quant::BLOCK], data: q.scales });
                outs.push(Out { name: format!("{stem}.weight_global_scale"), dtype: Dtype::F32, shape: vec![1], data: q.global_scale.to_le_bytes().to_vec() });
                n_req += 1;
            } else {
                for (n, v) in parts {
                    bytes_out += v.data().len() as u64;
                    outs.push(Out { name: n.clone(), dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() });
                    n_copy += 1;
                }
            }
            if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
        }
    }
    if shard_idx == 0 {
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join("model.safetensors")).expect("write");
    } else {
        write_shard(&mut outs, &mut weight_map, &mut shard_idx);
        let index = serde_json::json!({ "metadata": { "total_size": bytes_out }, "weight_map": weight_map });
        std::fs::write(outd.join("model.safetensors.index.json"), serde_json::to_string_pretty(&index).unwrap()).expect("write index");
    }
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json"] {
        let src = ind.join(f);
        if src.exists() { let _ = std::fs::copy(&src, outd.join(f)); }
    }
    println!("  requant-gdn done: {} GDN tensors re-quantized fp8->nvfp4, {} copied verbatim, {} shards, {:.1} GB",
             n_req, n_copy, shard_idx.max(1), bytes_out as f64 / 1e9);
}

/// `--requant-sim --model-dir <packed-dir> --out <dir> [--scope all|gate_up] [--shard-start N] [--shard-end M]`
///
/// E26 probe: bake the 2-bit error INTO an NVFP4 artifact's routed-expert tensors — dequant
/// NVFP4 → q2 round-trip (`fake_quant_q2`: per-16 block, Lloyd-Max levels, E4M3 scale) → requant
/// NVFP4. Bytes/format unchanged; the engine serves the result unmodified, so 2-bit quality is
/// measured on the REAL serving path at full speed before any 2-bit kernel exists. The draft
/// head's experts (mtp.*) stay NVFP4. Output preserves the input's shard layout + index, so runs
/// compose with the Python probe (shards already present in --out are skipped).
pub(crate) fn run_requant_sim(args: &[String]) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    use gb10_inference::quant;
    let ind_s = parse_arg(args, "--model-dir").expect("--requant-sim requires --model-dir <packed-dir>");
    let out_s = parse_arg(args, "--out").expect("--requant-sim requires --out <dir>");
    let scope = parse_arg(args, "--scope").unwrap_or("all");
    let shard_start: usize = parse_arg(args, "--shard-start").and_then(|s| s.parse().ok()).unwrap_or(1);
    let shard_end: usize = parse_arg(args, "--shard-end").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let ind = std::path::Path::new(ind_s);
    let outd = std::path::Path::new(out_s);
    std::fs::create_dir_all(outd).expect("create --out dir");
    println!("requant-sim: {} -> {} (scope {scope}, shards {shard_start}..{shard_end})", ind_s, out_s);

    let index_path = ind.join("model.safetensors.index.json");
    let shards: Vec<std::path::PathBuf> = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).expect("read index");
        let idx: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
        idx["weight_map"].as_object().unwrap().values().filter_map(|v| v.as_str())
            .collect::<std::collections::BTreeSet<_>>().into_iter().map(|s| ind.join(s)).collect()
    } else { vec![ind.join("model.safetensors")] };

    let meta = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("quant_recipe".to_string(), format!("sim({scope}): 2-bit RTN over nvfp4 — quality simulation, not a serving recipe")),
    ]);
    let is_expert = |n: &str| -> bool {
        if n.starts_with("mtp.") || !n.contains(".mlp.experts.") { return false; }
        if scope == "gate_up" && !n.contains("gate_up_proj") { return false; }
        n.ends_with("_packed")
    };

    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    let t0 = std::time::Instant::now();
    let (mut n_req, mut n_skip_shards) = (0usize, 0usize);
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
            if is_expert(n) {
                let get = |suffix: &str| -> &TensorView {
                    let full = format!("{stem}.weight{suffix}");
                    &tvec.iter().find(|(tn, _)| *tn == full)
                        .unwrap_or_else(|| panic!("{full} missing in {fname} (expert triple split across shards?)")).1
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
                let mut w = quant::dequantize_nvfp4(&q);
                quant::fake_quant_q2(&mut w, m, k);
                let q2 = quant::quantize_nvfp4(&w, m, k);
                outs.push(Out { name: n.clone(), dtype: Dtype::U8, shape: vec![m, kh], data: q2.qweight });
                outs.push(Out { name: format!("{stem}.weight_scale"), dtype: Dtype::F8_E4M3,
                                shape: vec![m, k / quant::BLOCK], data: q2.scales });
                outs.push(Out { name: format!("{stem}.weight_global_scale"), dtype: Dtype::F32,
                                shape: vec![1], data: q2.global_scale.to_le_bytes().to_vec() });
                n_req += 1;
                println!("    q2 {stem} [{m}x{k}]");
            } else if n.ends_with("_scale") {
                // The scale siblings ride out with their packed triple — never copy them alone.
                let packed_name = format!("{stem}.weight_packed");
                if is_expert(&packed_name) && tvec.iter().any(|(tn, _)| *tn == packed_name) {
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
        println!("    wrote {fname} ({} tensors, {:.1}s)", outs.len(), t_shard.elapsed().as_secs_f32());
    }
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json",
              "model.safetensors.index.json"] {
        let src = ind.join(f);
        if src.exists() && !outd.join(f).exists() { let _ = std::fs::copy(&src, outd.join(f)); }
    }
    println!("requant-sim done: {n_req} expert tensors q2'd, {n_skip_shards} shards skipped (already present), {:.1}s",
             t0.elapsed().as_secs_f32());
}

/// MTP end-to-end probe: runs full speculative decoding (draft → verify → accept → rollback) and
/// checks the output is token-for-token identical to sequential greedy (lossless), while reporting
/// the acceptance rate and speedup vs sequential.
/// `--quantize --model-dir <in> --out <dir> --recipe <spec>` — the offline quantizer.
///
/// Emits **compressed-tensors** layout, byte-compatible with HF (so our artifacts and theirs are
/// mutually loadable — and so the format is one we already validated against a real checkpoint):
///
/// ```text
///   NVFP4:  {name}.weight_packed        U8       [M, K/2]   nibble-packed E2M1
///           {name}.weight_scale         F8_E4M3  [M, K/16]  block scales
///           {name}.weight_global_scale  F32      [1]        (6*448)/amax — DIVIDE on dequant
///   FP8:    {name}.weight               F8_E4M3  [M, K]
///           {name}.weight_scale         F32      [M]        one per output row
///   else:   {name}.weight               copied through unchanged (norms, conv1d, A_log, dt_bias…)
/// ```
///
/// Recipe is the same syntax as the fake-quant knob, e.g. `all`, `all,gdn:fp8`, `all:fp8`.
/// Measured on 9B (held-out prose+code, bf16 PPL 7.622): `all` → 8.332 (+9.3%),
/// `all,gdn:fp8` → 8.036 (+5.4%), `all:fp8` → 7.673 (+0.7%).
/// `--df2-bake-nvfp4 | --df2-bake-fp8` `--draft-dir <src> --out <dst>` — PLAN/25 §1a/§1a-fp8:
/// weight-only bake of the DFlash2 drafter. Emits a baked artifact DIR (see
/// `dflash2::load::load_quantized` for the layout): `nvfp4.safetensors` (packed linears,
/// compressed-tensors convention) or `fp8.safetensors` (E4M3 bytes + per-row f32 scales) +
/// `model.safetensors` (the kept hi-prec tensors + the BF16 twins `prime_window` needs) +
/// `config.json` stamped with `df2_quant`. CPU-only; the serving round picks the dir up via
/// `--draft-dir` exactly like a BF16 artifact. Selector codebooks/hp/norms/base_kernels stay
/// BF16 per the plan (the walk gathers codebooks row-wise — quantizing them saves memory, not
/// bandwidth, and spends draft quality to do it).
pub(crate) fn run_df2_bake(args: &[String], fp8: bool) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    let src = gb10_inference::opt_or_arg(args, "--draft-dir").expect("--draft-dir <src artifact> required");
    // FLASHNEXT-DF2-NVFP4: the inventories follow the SOURCE artifact's geometry (its config.json),
    // not the 27B defaults — a missing/unreadable config is a hard stop, never a silent 27B bake.
    gb10_inference::dflash2::install_geom_from_dir(&src)
        .unwrap_or_else(|e| panic!("--draft-dir {src}: cannot read the artifact geometry (config.json): {e:#}"));
    let out_dir = parse_arg(args, "--out").expect("--out <baked dir> required");
    let outd = std::path::Path::new(&out_dir);
    std::fs::create_dir_all(outd).expect("create --out dir");

    let path = std::path::Path::new(&src).join("model.safetensors");
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(&raw);
    let src_sha: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let st = SafeTensors::deserialize(&raw).expect("parse src safetensors");

    let inv: std::collections::HashMap<String, Vec<usize>> =
        gb10_inference::dflash2::inventory().into_iter().map(|(n, s)| (n.to_string(), s.to_vec())).collect();
    let quant_inv = gb10_inference::dflash2::load::quantized_inventory();
    let quant_names: std::collections::HashSet<String> =
        quant_inv.iter().map(|(n, _, _)| n.clone()).collect();

    // Validate the bake set up front: every quantized name exists with M%16==0 plus the
    // repack's K constraint (fp4: K%32 for the paired nibbles; fp8: K%16), and the source is
    // the 81-tensor BF16 artifact.
    for (name, m, k) in &quant_inv {
        let shape = inv.get(name).unwrap_or_else(|| panic!("{name}: not in the artifact inventory"));
        assert!(shape.as_slice() == &[*m, *k], "{name}: shape {shape:?} != [{m},{k}]");
        if fp8 { assert!(m % 16 == 0 && k % 16 == 0, "{name}: {m}x{k} fails M%16/K%16"); }
        else   { assert!(m % 16 == 0 && k % 32 == 0, "{name}: {m}x{k} fails M%16/K%32"); }
    }

    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    let (mut quant_outs, mut kept_outs): (Vec<Out>, Vec<Out>) = (Vec::new(), Vec::new());
    let (mut bytes_q, mut bytes_kept) = (0u64, 0u64);

    // 1. Quantize the linears -> the sidecar safetensors (the --quantize emit convention).
    for (name, m, k) in &quant_inv {
        let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(view.dtype() == Dtype::BF16, "{name}: not BF16");
        assert!(view.shape() == &vec![*m, *k], "{name}: shape {:?}", view.shape());
        let w_cow = gb10_inference::podread::pod_cow::<half::bf16>(view.data()); // S-B9-REL-PPL: offsets may be unaligned
        let w: &[half::bf16] = &w_cow;
        if fp8 {
            let q = gb10_inference::quant::quantize_fp8(w, *m, *k);
            bytes_q += (q.qweight.len() + q.row_scale.len() * 4) as u64;
            quant_outs.push(Out { name: format!("{name}.weight_packed"), dtype: Dtype::U8,
                                  shape: vec![*m, *k], data: q.qweight });
            let mut rs = Vec::with_capacity(q.row_scale.len() * 4);
            for s in &q.row_scale { rs.extend_from_slice(&s.to_le_bytes()); }
            quant_outs.push(Out { name: format!("{name}.weight_row_scale"), dtype: Dtype::F32,
                                  shape: vec![*m], data: rs });
        } else {
            let q = gb10_inference::quant::quantize_nvfp4(w, *m, *k);
            bytes_q += (q.qweight.len() + q.scales.len() + 4) as u64;
            quant_outs.push(Out { name: format!("{name}.weight_packed"), dtype: Dtype::U8,
                                  shape: vec![*m, k / 2], data: q.qweight });
            quant_outs.push(Out { name: format!("{name}.weight_scale"), dtype: Dtype::F8_E4M3,
                                  shape: vec![*m, k / 16], data: q.scales });
            quant_outs.push(Out { name: format!("{name}.weight_global_scale"), dtype: Dtype::F32,
                                  shape: vec![1], data: q.global_scale.to_le_bytes().to_vec() });
        }
    }

    // 2. Copy the kept set -> model.safetensors: everything NOT quantized (codebooks, hp, norms,
    //    base_kernels) plus the BF16 twins prime_window needs (fc + per-layer k/v).
    let twins: std::collections::HashSet<String> =
        gb10_inference::dflash2::load::bf16_twin_inventory().into_iter().map(|(n, _, _)| n).collect();
    let mut kept_names: Vec<&String> = inv.keys()
        .filter(|n| !quant_names.contains(*n) || twins.contains(*n)).collect();
    kept_names.sort();
    for name in &kept_names {
        let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        bytes_kept += view.data().len() as u64;
        kept_outs.push(Out { name: (*name).clone(), dtype: view.dtype(),
                             shape: view.shape().to_vec(), data: view.data().to_vec() });
    }

    // 3. Write both safetensors + the stamped config.
    fn views(outs: &Vec<Out>) -> Vec<(String, TensorView<'_>)> {
        outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view")))
            .collect()
    }
    let sidecar = if fp8 { "fp8.safetensors" } else { "nvfp4.safetensors" };
    safetensors::serialize_to_file(views(&quant_outs), None, &outd.join(sidecar)).expect("write sidecar");
    safetensors::serialize_to_file(views(&kept_outs), None, &outd.join("model.safetensors")).expect("write kept");
    let cfg_raw = std::fs::read_to_string(std::path::Path::new(&src).join("config.json"))
        .unwrap_or_else(|e| panic!("read src config.json: {e}"));
    let mut cfg: serde_json::Value = serde_json::from_str(&cfg_raw)
        .unwrap_or_else(|e| panic!("parse src config.json: {e}"));
    cfg["df2_quant"] = serde_json::json!(if fp8 { "fp8" } else { "nvfp4" });
    cfg["df2_quant_source_sha256"] = serde_json::json!(src_sha);
    cfg["df2_quant_recipe"] = serde_json::json!(if fp8 { "fp8-weight-only-rtw" } else { "nvfp4-weight-only-rtw" });
    std::fs::write(outd.join("config.json"), serde_json::to_string_pretty(&cfg)
        .expect("serialize config")).expect("write config");

    let (fmt_name, quant_bits): (&str, u64) = if fp8 { ("fp8", 8) } else { ("nvfp4", 4) };
    println!("df2-bake-{}: {} -> {}", fmt_name, src, out_dir);
    println!("  source sha256:  {src_sha}");
    println!("  quantized:      {} linears, packed {:.2} GB at {quant_bits}-bit (was {:.2} GB bf16)",
             quant_inv.len(), bytes_q as f64 / 1e9,
             quant_inv.iter().map(|(_, m, k)| (m * k * 2) as u64).sum::<u64>() as f64 / 1e9);
    println!("  kept bf16:      {} tensors (hi-prec + prime twins), {:.2} GB", kept_names.len(),
             bytes_kept as f64 / 1e9);
    println!("  total on disk:  {:.2} GB (source artifact {:.2} GB)",
             (bytes_q + bytes_kept) as f64 / 1e9, raw.len() as f64 / 1e9);
}

/// `--df2-quant-fidelity --draft-dir <src>` — PLAN/25 §1a-fp8b kill-switch: quantize the
/// drafter's 46 block linears three ways (nvfp4 / fp8-per-row / fp8-block) and report the
/// host dequant rel-L2 vs bf16 per tensor group. CPU-only. Decision rule: fp8-block must be
/// ~3x cleaner than nvfp4's ~0.105 overall or the block-scale kernel is not worth building.
pub(crate) fn run_df2_quant_fidelity(args: &[String]) {
    use safetensors::SafeTensors;
    let src = gb10_inference::opt_or_arg(args, "--draft-dir").expect("--draft-dir <src artifact> required");
    gb10_inference::dflash2::install_geom_from_dir(&src)
        .unwrap_or_else(|e| panic!("--draft-dir {src}: cannot read the artifact geometry (config.json): {e:#}"));
    let path = std::path::Path::new(&src).join("model.safetensors");
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let st = SafeTensors::deserialize(&raw).expect("parse src safetensors");

    let rel_l2 = |orig: &[half::bf16], deq: &[f32]| -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (o, d) in orig.iter().zip(deq.iter()) {
            let diff = (o.to_f32() - *d) as f64;
            num += diff * diff;
            den += (*o).to_f32() as f64 * (*o).to_f32() as f64;
        }
        (num / den).sqrt()
    };

    let quant_inv = gb10_inference::dflash2::load::quantized_inventory();
    let (mut g_attn, mut g_mlp, mut g_all): ([f64; 3], [f64; 3], [f64; 3]) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    let (mut n_attn, mut n_mlp) = (0usize, 0usize);
    println!("df2-quant-fidelity: {src}");
    println!("{:<44} {:>10} {:>10} {:>10}", "tensor", "nvfp4", "fp8-row", "fp8-block");
    for (name, m, k) in &quant_inv {
        let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(view.dtype() == safetensors::Dtype::BF16, "{name}: not BF16");
        let w_cow = gb10_inference::podread::pod_cow::<half::bf16>(view.data()); // S-B9-REL-PPL: offsets may be unaligned
        let w: &[half::bf16] = &w_cow;
        let orig_f32: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();

        let q4 = gb10_inference::quant::quantize_nvfp4(w, *m, *k);
        let d4 = gb10_inference::quant::dequantize_nvfp4_f32(&q4);
        let l4 = rel_l2(w, &d4);

        let qr = gb10_inference::quant::quantize_fp8(w, *m, *k);
        let dr = gb10_inference::quant::dequantize_fp8(&qr);
        let lr = rel_l2(w, &dr.iter().map(|x| x.to_f32()).collect::<Vec<f32>>()[..]);

        let qb = gb10_inference::quant::quantize_fp8_block16(w, *m, *k);
        let db = gb10_inference::quant::dequantize_fp8_block16_f32(&qb);
        let lb = rel_l2(w, &db);

        let is_mlp = name.contains("mlp.");
        if is_mlp { g_mlp = [g_mlp[0]+l4*l4, g_mlp[1]+lr*lr, g_mlp[2]+lb*lb]; n_mlp += 1; }
        else { g_attn = [g_attn[0]+l4*l4, g_attn[1]+lr*lr, g_attn[2]+lb*lb]; n_attn += 1; }
        g_all = [g_all[0]+l4*l4, g_all[1]+lr*lr, g_all[2]+lb*lb];
        println!("{:<44} {:>10.4} {:>10.4} {:>10.4}", name, l4, lr, lb);
    }
    let rms = |g: [f64; 3], n: usize| [g[0], g[1], g[2]].map(|v| (v / n as f64).sqrt());
    let (a, mlp, all) = (rms(g_attn, n_attn), rms(g_mlp, n_mlp), rms(g_all, quant_inv.len()));
    println!("\nper-tensor RMS rel-L2 over {} attention / {} MLP linears:", n_attn, n_mlp);
    println!("  attention: nvfp4 {:.4}  fp8-row {:.4}  fp8-block {:.4}", a[0], a[1], a[2]);
    println!("  MLP:       nvfp4 {:.4}  fp8-row {:.4}  fp8-block {:.4}", mlp[0], mlp[1], mlp[2]);
    println!("  ALL:       nvfp4 {:.4}  fp8-row {:.4}  fp8-block {:.4}", all[0], all[1], all[2]);
    let ratio = all[0] / all[2];
    println!("\nfp8-block vs nvfp4: {:.2}x cleaner -> {}", ratio,
             if ratio >= 2.5 { "GO (block-scale kernel justified)" } else { "NO-GO (not enough cleaner to matter)" });
}
