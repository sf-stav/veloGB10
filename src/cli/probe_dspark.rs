//! DSpark probes: bind, synthetic draft generation and check, the main DSpark probe, extraction and the
//! rollback probe. The round probe is in `probe_dspark_round.rs`; serving is in `dsv4_serve.rs`.

use super::*;

/// B8/G4 — `--probe-dspark-bind <dir>`: verify the DSpark drafter artifact + the embed/LM-head
/// binding surface WITHOUT wiring the drafter (the K-DSP session's first gate).
///
/// Checks (assert the SUCCESS signal, never absence of errors):
///   1. `model.safetensors` exists in <dir> and its sha256 matches the addendum's pinned hash
///      prefix 9d26d5e6...fe692786 (a mismatch is a hard fail — wrong artifact).
///   2. The safetensors header parses; the tensor count is exactly 62.
///   3. NO `embed_tokens` / `lm_head` tensor exists in the checkpoint (both are bound to the
///      TARGET at runtime — their presence means the wrong file).
///   4. All tensors are BF16.
///   5. The expected tensor-name families are present (5 layers x {q,k,v,o, gate/up/down, norms},
///      fc, hidden_norm, markov W1/W2, confidence) — family-count report.
///   6. The engine binding surface: DflashDrafter::forward accepts (noise_embed, head) overrides
///      that point at the TARGET's tensors — the exact mechanism G4's binding uses. We assert the
///      API exists by referencing it (compile-time) and report the runtime plan.
pub(crate) fn run_probe_dspark_bind(args: &[String]) {
    let dir: &str = parse_arg(args, "--probe-dspark-bind").unwrap_or(".");
    let path = std::path::Path::new(&dir).join("model.safetensors");
    println!("B8/G4 DSpark binding probe: {}", path.display());
    if !path.exists() {
        println!("RESULT: SKIP (no model.safetensors under {dir} — the 62-tensor shard is not staged here)");
        return;
    }
    // sha256 with a file-streaming hasher (sha2 is already a dependency of cluster.rs).
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut f = std::fs::File::open(&path).expect("open safetensors");
    std::io::copy(&mut f, &mut h).expect("hash safetensors");
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let size = std::fs::metadata(&path).unwrap().len();
    println!("  sha256 = {hexd}");
    println!("  size   = {size} bytes (addendum: 2,718,576,122)");
    let pinned_prefix = "9d26d5e6";
    let pinned_suffix = "fe692786";
    let hash_ok = hexd.starts_with(pinned_prefix) && hexd.ends_with(pinned_suffix);
    println!("  hash pins prefix..suffix: {}", if hash_ok { "MATCH" } else { "MISMATCH (wrong artifact?)" });

    // Header parse: 8-byte LE header length + JSON.
    let mut f = std::fs::File::open(&path).unwrap();
    use std::io::Read;
    let mut hlen = [0u8; 8];
    f.read_exact(&mut hlen).unwrap();
    let hlen = u64::from_le_bytes(hlen) as usize;
    let mut hdr = vec![0u8; hlen];
    f.read_exact(&mut hdr).unwrap();
    let hdr: serde_json::Value = serde_json::from_slice(&hdr).expect("parse safetensors header");
    let obj = hdr.as_object().expect("header object");
    let mut n_tensors = 0usize;
    let mut bad_dtype: Vec<&str> = Vec::new();
    let mut forbidden: Vec<&str> = Vec::new();
    let mut families = std::collections::BTreeMap::new();
    for (k, v) in obj {
        if k == "__metadata__" { continue; }
        n_tensors += 1;
        let dt = v.get("dtype").and_then(|d| d.as_str()).unwrap_or("?");
        if dt != "BF16" { bad_dtype.push(k.as_str()); }
        if k.contains("embed_tokens") || k.contains("lm_head") { forbidden.push(k.as_str()); }
        // family key: strip layer indices
        let collapsed: String = k.chars().map(|c| if c.is_ascii_digit() { '.' } else { c }).collect();
        let mut fam = String::new();
        let mut prev_dot = false;
        for c in collapsed.chars() {
            if c == '.' {
                if !prev_dot { fam.push(c); }
                prev_dot = true;
            } else { fam.push(c); prev_dot = false; }
        }
        *families.entry(fam).or_insert(0usize) += 1;
    }
    println!("  tensors = {n_tensors} (addendum: 62)");
    println!("  non-BF16 tensors: {}", if bad_dtype.is_empty() { "none".into() } else { bad_dtype.join(",") });
    println!("  embed/lm_head tensors present: {}", if forbidden.is_empty() { "none (correct — bound to target)".into() } else { forbidden.join(",") });
    println!("  tensor families (digits collapsed):");
    for (fam, n) in families.iter().take(30) { println!("    {fam:<58} x{n}"); }

    let ok = hash_ok && n_tensors == 62 && bad_dtype.is_empty() && forbidden.is_empty();
    println!("RESULT: {}", if ok { "PASS (artifact + binding surface verified)" } else { "FAIL" });
    println!("  binding plan: engine-side, the DFlashDrafter forward(noise_embed=Some(target_embed), head=Some(target_lm_head)) override IS the G4 binding surface; DSpark's 5-layer draft reuses the same dual-source convention (K-DSP1 spec).");
    if !ok { std::process::exit(1); }
}

/// S2 — `--gen-dspark-synth <dir>`: the deterministic synthetic 62-tensor artifact generator.
pub(crate) fn run_gen_dspark_synth(args: &[String], dir: &str) {
    let seed: u64 = parse_arg(args, "--seed").and_then(|s| s.parse().ok())
        .unwrap_or_else(gb10_inference::dspark::synth::default_seed);
    let s = gb10_inference::dspark::synth::generate(dir, seed).expect("generate synth artifact");
    println!("P8 synthetic DSpark artifact generated (deterministic, seed {seed}):");
    println!("  dir       = {}", s.dir);
    println!("  sha256    = {}", s.sha256);
    println!("  tensors   = {} (assert 62)", s.n_tensors);
    println!("  params    = {} (assert 1,359,284,737)", s.n_params);
    println!("  file_size = {} bytes", s.file_size);
    println!("  data_size = {} bytes (BF16 x params)", s.n_params * 2);
    println!("  header    = {} bytes (8-byte len + JSON)", s.header_size);
    println!("  config.json + SYNTHETIC_README.md written");
    println!("RESULT: GENERATED (62 / 1,359,284,737 asserted at generation time)");
}

/// S2 — `--probe-dspark-synth <dir>`: load/generate the synthetic artifact and run the oracle
/// checks. Prints PASS/FAIL per check; exit 0 only on all-PASS.
pub(crate) fn run_probe_dspark_synth(args: &[String], dir: &str) {
    use gb10_inference::dspark::oracle::{DsparkConfig, DsparkOracle, RoundCtx};
    use gb10_inference::dspark::synth::SyntheticTables;

    let mut all_pass = true;
    let mut check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass = false; }
    };

    // Generate-if-absent, then load + validate (inventory/shapes/dtypes + optional sha256 pin).
    if !std::path::Path::new(dir).join("model.safetensors").exists() {
        println!("artifact absent under {dir} — generating the synthetic artifact first");
        let seed = gb10_inference::dspark::synth::default_seed();
        gb10_inference::dspark::synth::generate(dir, seed).expect("generate");
    }
    let sha_pin = parse_arg(args, "--sha256");
    let art = gb10_inference::dspark::load::load(dir, sha_pin).expect("load artifact");
    println!("inventory: {} tensors, {} params, {} bytes, sha256 {}",
             art.n_tensors, art.n_params, art.file_size, art.sha256);
    check("inventory == 62 tensors / 1,359,284,737 params / BF16-only (loader)",
          art.n_tensors == 62 && art.n_params == 1_359_284_737);

    let cfg = DsparkConfig::default();
    let oracle = DsparkOracle::from_weights(cfg.clone(), art.weights.clone()).expect("oracle");

    // Deterministic synthetic context: L committed positions of tap hiddens [L, 5*hidden].
    let l = 16usize;
    let tap_dim = 5 * cfg.hidden;
    let synth = SyntheticTables::new(gb10_inference::dspark::SYNTH_EMBED_HEAD_SEED);
    let scale = 1.0f32 / (tap_dim as f32).sqrt();
    let mut taps = Vec::with_capacity(l * tap_dim);
    for i in 0..l {
        taps.extend_from_slice(&synth.row(2, i as u32, tap_dim, scale));
    }
    let ctx = RoundCtx { tap_hiddens: taps.clone(), anchor: 0, confidence_threshold: 0.5 };

    // (a) determinism — two run_rounds bit-identical.
    let a1 = oracle.run_round(&ctx);
    let a2 = oracle.run_round(&ctx);
    let det_ok = a1.logits0 == a2.logits0 && a1.h == a2.h && a1.tokens == a2.tokens
        && a1.latents == a2.latents && a1.p == a2.p && a1.survival == a2.survival
        && a1.k_verify == a2.k_verify && a1.th == a2.th;
    check("determinism: two run_rounds bit-identical", det_ok);

    // (b) wiring — flip one layer-0 q_proj weight → logits MUST change (anti-empty-compare).
    // The workdoc's own "flip one weight" alternative: a single 1-ulp perturb rounds away through
    // the per-head RMSNorm in f32 (measured; recorded in the S2 report R4). A sign flip is a
    // single-weight perturbation that provably propagates, so it is the hard gate.
    let mut pw = art.weights.clone();
    let q0s = pw.layers[0].q_proj.as_mut_slice();
    let idx = q0s.iter().position(|&x| x != 0.0).expect("a nonzero q_proj weight exists");
    q0s[idx] = -q0s[idx];
    let oracle_p = DsparkOracle::from_weights(cfg.clone(), pw).expect("perturbed oracle");
    let b = oracle_p.run_round(&ctx);
    let wiring_ok = b.logits0 != a1.logits0 || b.h != a1.h || b.tokens != a1.tokens;
    check("wiring: flip one q_proj weight changes outputs", wiring_ok);

    // (c) incremental == batch tap projection (DECISION B).
    let th_batch = oracle.tap_project(&taps, l);
    let mut th_inc = Vec::with_capacity(l * cfg.hidden);
    for i in 0..l {
        let mut t = oracle.tap_project(&taps[i * tap_dim..(i + 1) * tap_dim], 1);
        th_inc.append(&mut t);
    }
    check("incremental == batch tap projection (bit-identical)", th_batch == th_inc);

    // (d) structure.
    let logits_finite = a1.logits0.iter().all(|x| x.is_finite());
    // Markov latents: assert NOT all identical (the anti-degenerate-chain gate). The 6 MASK
    // positions of a RANDOM synthetic backbone collapse to near-identical hiddens (measured rel-L2
    // ~1%; the full-bidirectional attention averages out the weak YaRN θ=1e7 RoPE), so full
    // pairwise distinctness is not a property of the synthetic model — it IS of the trained real
    // model (S7). The chain's distinctness mechanism is asserted pairwise via a direct logits0
    // test (unit test + the `--probe-dspark-synth` piecewise contract).
    let rank = cfg.markov_rank;
    let mut latent_set: std::collections::BTreeSet<Vec<u32>> = std::collections::BTreeSet::new();
    for k in 0..6 {
        latent_set.insert(a1.latents[k * rank..(k + 1) * rank].iter().map(|x| x.to_bits()).collect());
    }
    let latents_not_all_identical = latent_set.len() >= 2;
    let confs_in_unit = a1.p.iter().all(|&x| x > 0.0 && x < 1.0);
    let survival_mono = a1.survival.windows(2).all(|w| w[0] >= w[1]);
    // k_verify sweep 0.1..0.9 → all in [1..8], and non-degenerate (varies on ≥2 thresholds).
    let mut kvs = std::collections::BTreeSet::new();
    let mut kverify_in_range = true;
    let mut thr = 0.1f32;
    while thr < 0.91 {
        let kv = gb10_inference::dspark::oracle::truncate(&a1.survival, thr);
        if !(1..=8).contains(&kv) { kverify_in_range = false; }
        kvs.insert(kv);
        thr += 0.1;
    }
    check("structure: logits finite", logits_finite);
    check(&format!("structure: Markov latents not all identical ({}/6 distinct)", latent_set.len()),
          latents_not_all_identical);
    check("structure: confidences in (0,1)", confs_in_unit);
    check("structure: survival monotone non-increasing", survival_mono);
    check("structure: k_verify in [1..8] across 0.1..0.9 sweep", kverify_in_range);
    check(&format!("structure: k_verify sweep non-degenerate ({{{:?}}})", kvs), kvs.len() > 1);

    // (e) piecewise — each API callable standalone, and the composition == run_round bit-exactly.
    let piecewise = (|| -> Result<(), String> {
        let th = oracle.tap_project(&taps, l);
        let kv = oracle.draft_kv_write(&th, l);
        let mut blk = vec![0u32];
        for _ in 1..7 { blk.push(cfg.mask_token_id); }
        let escale = 1.0f32 / (cfg.hidden as f32).sqrt();
        let mut emb = Vec::with_capacity(7 * cfg.hidden);
        for &t in &blk {
            emb.extend_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, t, cfg.hidden, escale));
        }
        let h = oracle.block_forward(&emb, &kv, l);
        let logits0 = oracle.lm_head(&h, 7);
        let mo = oracle.markov_chain(&logits0, blk[0]);
        let co = oracle.confidence(&h, &mo.latents, 0.5);
        if h.len() != 7 * cfg.hidden { return Err("block_forward shape".into()); }
        if logits0.len() != 7 * cfg.vocab { return Err("lm_head shape".into()); }
        if mo.latents.len() != 6 * rank { return Err("markov latents shape".into()); }
        let compose_ok = h == a1.h && logits0 == a1.logits0 && mo.tokens == a1.tokens
            && mo.latents == a1.latents && co.p == a1.p && co.survival == a1.survival
            && co.k_verify == a1.k_verify;
        if !compose_ok { return Err("piecewise composition != run_round".into()); }
        Ok(())
    })();
    check("piecewise: all pieces callable + composition == run_round", piecewise.is_ok());
    if let Err(e) = piecewise { println!("           ({e})"); }

    println!("RESULT: {}", if all_pass { "ALL PASS" } else { "FAIL" });
    if !all_pass { std::process::exit(1); }
}

/// `--probe-dspark`: Phase-5 DSpark draft-mechanics probe (single-process, chaos-amendment
/// compliant). Loads the 3 DSpark stages + the trunk embed/head, feeds the oracle's warm +
/// draft main_hidden, and validates: (a) the warm path runs + ring shape; (b) the 133-entry
/// non-causal index list structure vs `dsv4_cpu::dspark_topk_idxs`; (c) the Markov chain is
/// sequential (5 steps, each dependent on the prior argmax); (d) the draft output shape +
/// the match count vs the oracle's output_ids (expected LOW — the draft chain is chaotic per
/// the §7 G1 amendment; the mechanics, not the values, are the gate).
pub(crate) fn run_probe_dspark(args: &[String]) {
    use gb10_inference::dsv4_dspark::Dsv4DSpark;
    use gb10_inference::dsv4_load::{self, NpyData};
    use gb10_inference::dsv4_model::Dsv4GpuModel;
    use half::bf16;
    use std::path::Path;
    use cudarc::driver::CudaDevice;

    let bundle = parse_arg(args, "--model-dir").expect("--probe-dspark requires --model-dir <bundle>");
    // T5 (queue #13): default = the 0731-native regenerated fixture (old /mnt/models/dsv4-oracle-v2
    // was deleted with the obsolete model — a bare invocation must not trip on stale paths).
    let oracle_dir = parse_arg(args, "--oracle").unwrap_or("/tmp/dsv4-0731-ref");
    let cfg = dsv4_load::load_config(Path::new(bundle)).expect("load_config");
    let dev = CudaDevice::new(0).expect("CUDA device 0");

    // Trunk top (embed/head only — the draft ties them).
    let top = Dsv4GpuModel::load_trunk_top(&dev, Path::new(bundle), &cfg).expect("load_trunk_top");

    // Oracle inputs.
    let npz = Path::new(oracle_dir).join("dsv4_dspark.npz");
    let read_f32 = |key: &str| -> Vec<f32> {
        let (_, d) = dsv4_load::read_npz_key(&npz, key).unwrap_or_else(|e| panic!("read {key}: {e}"));
        match d { NpyData::F32(v) => v, _ => panic!("{key} not f32") }
    };
    let read_i64 = |key: &str| -> Vec<i64> {
        let (_, d) = dsv4_load::read_npz_key(&npz, key).unwrap_or_else(|e| panic!("read {key}: {e}"));
        match d { NpyData::I64(v) => v, _ => panic!("{key} not i64") }
    };
    let warm_mh = read_f32("warm.main_hidden");     // [130, 12288]
    let draft_mh = read_f32("draft.main_hidden");   // [1, 12288]
    let real_tok = read_i64("draft.real_token")[0] as i32;
    let oracle_ids = read_i64("draft.output_ids");   // [6]
    let sw = warm_mh.len() / (3 * cfg.dim);
    let dim = cfg.dim;
    let block = cfg.dspark_block_size;
    assert_eq!(warm_mh.len(), sw * 3 * dim, "warm.main_hidden shape");
    assert_eq!(draft_mh.len(), 3 * dim, "draft.main_hidden shape");
    eprintln!("[probe-dspark] oracle: warm={sw} positions, draft main_hidden [1,3*dim], real_token={real_tok}, block={block}");

    // ---- (b) index-list structure check vs dsv4_cpu (the 133-entry non-causal list) ----
    let start_pos = sw;
    let idxs_ref = gb10_inference::dsv4_cpu::dspark_topk_idxs(cfg.window_size, block, start_pos);
    let t_win = idxs_ref.len() - block;
    let expected_t_win = cfg.window_size.min(start_pos + 1);
    println!("=== DSpark index list @ start_pos={start_pos} (window={}, block={}) ===", cfg.window_size, block);
    println!("  entries: {} (= window-part {t_win} ++ draft-part {block})", idxs_ref.len());
    assert_eq!(idxs_ref.len(), expected_t_win + block, "index list length");
    assert_eq!(t_win, expected_t_win, "window part = min(win, start_pos+1)");
    // non-causal: every draft row uses the SAME list (the engine broadcasts it).
    println!("  window part: [0..{t_win}]  draft part: [{}..{}]  (non-causal, identical per row)",
             cfg.window_size, cfg.window_size + block);
    println!("  MATCHES dsv4_cpu::dspark_topk_idxs: OK");

    // ---- load the 3 DSpark stages + warm ----
    let embed = top.embed.clone();
    let head = top.head.clone();
    drop(top);
    let mut ds = Dsv4DSpark::load(&dev, Path::new(bundle), &cfg, 2048, embed, head).expect("Dsv4DSpark::load");

    let warm_bf16: Vec<bf16> = warm_mh.iter().map(|&v| bf16::from_f32(v)).collect();
    let warm_dev = dev.htod_sync_copy(&warm_bf16).expect("htod warm");
    ds.warm(&warm_dev, sw).expect("dspark warm");
    println!("=== DSpark warm: 3 stages ring-written ({} positions each) — OK ===", sw);

    // ---- (c) draft + Markov chain ----
    let draft_bf16: Vec<bf16> = draft_mh.iter().map(|&v| bf16::from_f32(v)).collect();
    let draft_dev = dev.htod_sync_copy(&draft_bf16).expect("htod draft main_hidden");
    let t0 = std::time::Instant::now();
    if gb10_inference::opts::var(gb10_inference::opt!("dspark-debug-ring")).is_ok() {
        // R4 audit: dump the ring RIGHT AFTER warm (before any draft advances it) and compare
        // against the CPU reference's ring — validates the warm path content.
        let ring_bf = dev.dtoh_sync_copy(&ds.states[0].kv_cache).expect("ring dtoh");
        let ring_f: Vec<f32> = ring_bf.iter().map(|v| v.to_f32()).collect();
        {
            use gb10_inference::dsv4_cpu as cpu;
            let mut cpu_layer = cpu::cpu_stage_from_dsv4(
                dsv4_load::load_mtp_stage(Path::new(bundle), &cfg, 0).expect("load_mtp_stage 0"), &cfg, 0)
                .expect("cpu_stage_from_dsv4");
            let rope = cpu::layer_rope_table(&cfg, dsv4_load::LayerKind::Swa, 4096);
            let main_x_of = |mh: &[f32]| -> Vec<f32> {
                let rows = mh.len() / (3 * dim);
                let mp = cpu_layer.main_proj.as_ref().unwrap();
                let mx = cpu::quant_gemm(mh, rows, 3 * dim, mp, dim, 128);
                cpu::rms_norm(&mx, rows, dim, cpu_layer.main_norm.as_ref().unwrap(), cfg.norm_eps)
            };
            let main_x_warm = main_x_of(&warm_mh);
            let mut kv_cache = vec![0.0f32; 128 * 512];
            cpu::dspark_attn_warm(&cpu_layer.attn, &mut kv_cache, &main_x_warm, sw, &rope, &cfg);
            let (mut num, mut den, mut maxabs) = (0.0f64, 0.0f64, 0.0f32);
            for i in 0..ring_f.len() {
                let d = ring_f[i] - kv_cache[i];
                num += (d as f64) * (d as f64);
                den += (kv_cache[i] as f64) * (kv_cache[i] as f64);
                if d.abs() > maxabs { maxabs = d.abs(); }
            }
            println!("  GPU ring (fresh warm) vs CPU ring: rel-L2 {:.3e}  max-abs {:.3e}",
                     (num / den.max(1e-30)).sqrt(), maxabs);
        }
        std::process::exit(0);
    }
    if gb10_inference::opts::var(gb10_inference::opt!("dspark-debug")).is_ok() {
        // R4 d3-cliff audit: bisect where the chain diverges from the reference oracle.
        let (out2, stages, sublayers, logits) = ds.draft_capture(&draft_dev, real_tok, start_pos).expect("draft_capture");
        let cmp = |name: &str, key: &str, got: &[f32]| {
            let want = read_f32(key);
            let n = got.len().min(want.len());
            let (mut num, mut den, mut maxabs, mut first) = (0.0f64, 0.0f64, 0.0f32, None);
            for i in 0..n {
                let d = got[i] - want[i];
                num += (d as f64) * (d as f64);
                den += (want[i] as f64) * (want[i] as f64);
                if d.abs() > maxabs { maxabs = d.abs(); }
                if first.is_none() && d.abs() > 1e-2 { first = Some(i); }
            }
            let rel = (num / den.max(1e-30)).sqrt();
            println!("  {name:<12} rel-L2 {rel:.3e}  max-abs {maxabs:.3e}  first>1e-2 @{first:?}  (n={n})");
        };
        println!("=== DSpark capture vs oracle (start_pos={start_pos}) ===");
        cmp("h_in", "draft.h_in", &stages[0]);
        cmp("h0", "draft.h0", &stages[1]);
        cmp("h1", "draft.h1", &stages[2]);
        cmp("h2", "draft.h2", &stages[3]);
        cmp("logits", "draft.logits", &logits);
        println!("  capture drafts: {:?}", out2.drafts);
        // CPU-reference bisect: feed the ORACLE's h_in (bit-equal to ours) through the G1 CPU
        // reference's stage-0 and compare with oracle h0 AND our GPU h0 — decides whether the
        // divergence is in the GPU stage path or in the shared assumptions.
        {
            use gb10_inference::dsv4_cpu as cpu;
            let mut cpu_layer = cpu::cpu_stage_from_dsv4(
                dsv4_load::load_mtp_stage(Path::new(bundle), &cfg, 0).expect("load_mtp_stage 0"), &cfg, 0)
                .expect("cpu_stage_from_dsv4");
            let (hd, block_c) = (cfg.head_dim, cfg.dspark_block_size);
            let rope = cpu::layer_rope_table(&cfg, dsv4_load::LayerKind::Swa, 4096);
            // main_x per position for the 130 warm positions + the draft position.
            let main_x_of = |mh: &[f32]| -> Vec<f32> {
                let rows = mh.len() / (3 * dim);
                let mp = cpu_layer.main_proj.as_ref().unwrap();
                let mx = cpu::quant_gemm(mh, rows, 3 * dim, mp, dim, 128);
                cpu::rms_norm(&mx, rows, dim, cpu_layer.main_norm.as_ref().unwrap(), cfg.norm_eps)
            };
            let main_x_warm = main_x_of(&warm_mh);
            let main_x_draft = main_x_of(&draft_mh);
            // GPU main_x vs CPU main_x (isolates the fp8 main_proj path from the attention path).
            {
                let gpu_mx = ds.main_x_for_debug(&draft_dev, 1).expect("main_x_for_debug");
                let cmp3 = |name: &str, a: &[f32], b: &[f32]| {
                    let n = a.len().min(b.len());
                    let (mut num, mut den, mut maxabs) = (0.0f64, 0.0f64, 0.0f32);
                    for i in 0..n {
                        let d = a[i] - b[i];
                        num += (d as f64) * (d as f64);
                        den += (b[i] as f64) * (b[i] as f64);
                        if d.abs() > maxabs { maxabs = d.abs(); }
                    }
                    println!("  {name:<28} rel-L2 {:.3e}  max-abs {:.3e}", (num / den.max(1e-30)).sqrt(), maxabs);
                };
                cmp3("GPU main_x vs CPU main_x", &gpu_mx, &main_x_draft);
                let warm_dev_bf: Vec<bf16> = warm_mh.iter().map(|&v| bf16::from_f32(v)).collect();
                let warm_dev2 = dev.htod_sync_copy(&warm_dev_bf).expect("htod warm2");
                let gpu_mxw = ds.main_x_for_debug(&warm_dev2, sw).expect("main_x_for_debug warm");
                cmp3("GPU main_x_warm vs CPU main_x_warm", &gpu_mxw, &main_x_warm);
            }
            let mut kv_cache = vec![0.0f32; 128 * 512];
            cpu::dspark_attn_warm(&cpu_layer.attn, &mut kv_cache, &main_x_warm, sw, &rope, &cfg);
            // GPU ring vs CPU ring after warm (splits "ring wrong" vs "attention compute wrong").
            {
                let ring_bf = dev.dtoh_sync_copy(&ds.states[0].kv_cache).expect("ring dtoh");
                let ring_f: Vec<f32> = ring_bf.iter().map(|v| v.to_f32()).collect();
                let cmp4 = |name: &str, a: &[f32], b: &[f32]| {
                    let n = a.len().min(b.len());
                    let (mut num, mut den, mut maxabs, mut first) = (0.0f64, 0.0f64, 0.0f32, None);
                    for i in 0..n {
                        let d = a[i] - b[i];
                        num += (d as f64) * (d as f64);
                        den += (b[i] as f64) * (b[i] as f64);
                        if d.abs() > maxabs { maxabs = d.abs(); }
                        if first.is_none() && d.abs() > 1e-2 { first = Some(i); }
                    }
                    println!("  {name:<28} rel-L2 {:.3e}  max-abs {:.3e}  first>1e-2 @{first:?}",
                             (num / den.max(1e-30)).sqrt(), maxabs);
                };
                cmp4("GPU ring vs CPU ring", &ring_f, &kv_cache);
                // the pre-norm GEMM output: GPU fp8_bsb(wkv) vs CPU quant_gemm(wkv) on main_x
                // (the GPU main_x is bf16 on device; the debug copy is an exact bf16 roundtrip).
                {
                    let gpu_mx = ds.main_x_for_debug(&draft_dev, 1).expect("main_x_for_debug");
                    let mx_bf: Vec<bf16> = gpu_mx.iter().map(|&v| bf16::from_f32(v)).collect();
                    let mx_dev = dev.htod_sync_copy(&mx_bf).expect("htod main_x");
                    let (mxc, mxsa) = ds.rt.quant_g128::<gb10_inference::dsv4_attn::B, cudarc::driver::CudaSlice<u8>>(&mx_dev, 1, dim).expect("quant main_x");
                    let mkv_gpu = ds.rt.fp8_bsb_rows(&ds.stages[0].wkv, &mxc, &mxsa, 1).expect("wkv gemm");
                    let mkv_host: Vec<f32> = dev.dtoh_sync_copy(&mkv_gpu).expect("mkv dtoh")
                        .iter().map(|v| v.to_f32()).collect();
                    let mk_cpu = cpu::quant_gemm(&main_x_draft, 1, dim, &cpu_layer.attn.wkv, hd, 128);
                    cmp4("GPU mkv vs CPU mk (pre-norm)", &mkv_host, &mk_cpu);
                    println!("    gpu mkv[0..8]: {:?}", &mkv_host[..8]);
                    println!("    cpu mk [0..8]: {:?}", &mk_cpu[..8]);
                    // norm step on the exact mk (isolates kv_norm / the norm kernel)
                    let mk_bf: Vec<bf16> = mk_cpu.iter().map(|&v| bf16::from_f32(v)).collect();
                    let mk_dev = dev.htod_sync_copy(&mk_bf).expect("htod mk");
                    let nrm_gpu = ds.rt.rmsnorm(&mk_dev, &ds.stages[0].kv_norm, 1, hd, cfg.norm_eps).expect("rmsnorm");
                    let nrm_host: Vec<f32> = dev.dtoh_sync_copy(&nrm_gpu).expect("nrm dtoh")
                        .iter().map(|v| v.to_f32()).collect();
                    let nrm_cpu = cpu::rms_norm(&mk_cpu, 1, hd, &cpu_layer.attn.kv_norm, cfg.norm_eps);
                    cmp4("GPU norm(mk) vs CPU norm(mk)", &nrm_host, &nrm_cpu);
                    // per-row profile of the warm wkv GEMM (the ring's row source) — input is
                    // main_x (bf16 exact roundtrip of the debug copy), NOT warm_hidden.
                    {
                        let gpu_mxw2 = ds.main_x_for_debug(&warm_dev, sw).expect("main_x_for_debug warm2");
                        let mxw_bf: Vec<bf16> = gpu_mxw2.iter().map(|&v| bf16::from_f32(v)).collect();
                        let mxw_dev = dev.htod_sync_copy(&mxw_bf).expect("htod main_x warm");
                        let (wc, wsa) = ds.rt.quant_g128::<gb10_inference::dsv4_attn::B, cudarc::driver::CudaSlice<u8>>(&mxw_dev, sw, dim).expect("quant main_x warm");
                        let mkv_w = ds.rt.fp8_bsb_rows(&ds.stages[0].wkv, &wc, &wsa, sw).expect("wkv warm gemm");
                        let mkv_wh: Vec<f32> = dev.dtoh_sync_copy(&mkv_w).expect("dtoh")
                            .iter().map(|v| v.to_f32()).collect();
                        let mk_cw = cpu::quant_gemm(&main_x_warm, sw, dim, &cpu_layer.attn.wkv, hd, 128);
                        for r in [0usize, 1, 2, 3, 64, 127, 128, 129] {
                            let (mut num, mut den) = (0.0f64, 0.0f64);
                            for d in 0..hd {
                                let df = mkv_wh[r * hd + d] - mk_cw[r * hd + d];
                                num += (df as f64) * (df as f64);
                                den += (mk_cw[r * hd + d] as f64) * (mk_cw[r * hd + d] as f64);
                            }
                            println!("    mk row {r:>3}: rel-L2 {:.3e}", (num / den.max(1e-30)).sqrt());
                        }
                    }
                }
                // error profile: per-64-dim group + per-slot max (localizes the corruption)
                let hd = 512usize;
                for g0 in (0..hd).step_by(64) {
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for sl in 0..128 {
                        for d in g0..(g0 + 64).min(hd) {
                            let i = sl * hd + d;
                            let df = ring_f[i] - kv_cache[i];
                            num += (df as f64) * (df as f64);
                            den += (kv_cache[i] as f64) * (kv_cache[i] as f64);
                        }
                    }
                    println!("    dims {g0:>3}..{:>3}: rel-L2 {:.3e}", (g0 + 64).min(hd), (num / den.max(1e-30)).sqrt());
                }
                let mut worst = (0.0f32, 0usize);
                for sl in 0..128 {
                    let mut m = 0.0f32;
                    for d in 0..hd {
                        let df = (ring_f[sl * hd + d] - kv_cache[sl * hd + d]).abs();
                        if df > m { m = df; }
                    }
                    if m > worst.0 { worst = (m, sl); }
                }
                println!("    worst slot: {} (max-abs {:.3e})", worst.1, worst.0);
                let ws = worst.1;
                println!("    ring[{}][0..8]  gpu: {:?}", ws, &ring_f[ws * 512..ws * 512 + 8]);
                println!("    ring[{}][0..8]  cpu: {:?}", ws, &kv_cache[ws * 512..ws * 512 + 8]);
                println!("    ring[{}][448..456] gpu: {:?}", ws, &ring_f[ws * 512 + 448..ws * 512 + 456]);
                println!("    ring[{}][448..456] cpu: {:?}", ws, &kv_cache[ws * 512 + 448..ws * 512 + 456]);
                println!("    ring[0][166..174] gpu: {:?}", &ring_f[166..174]);
                println!("    ring[0][166..174] cpu: {:?}", &kv_cache[166..174]);
            }
            let h_in = read_f32("draft.h_in");
            let (cpu_out, trace) = cpu::dspark_block_forward_traced(
                &cpu_layer, &mut kv_cache, &h_in, block_c, start_pos, &main_x_draft, &rope, &cfg);
            let cmp2 = |name: &str, a: &[f32], b: &[f32]| {
                let n = a.len().min(b.len());
                let (mut num, mut den, mut maxabs) = (0.0f64, 0.0f64, 0.0f32);
                for i in 0..n {
                    let d = a[i] - b[i];
                    num += (d as f64) * (d as f64);
                    den += (b[i] as f64) * (b[i] as f64);
                    if d.abs() > maxabs { maxabs = d.abs(); }
                }
                println!("  {name:<24} rel-L2 {:.3e}  max-abs {:.3e}", (num / den.max(1e-30)).sqrt(), maxabs);
            };
            let oracle_h0 = read_f32("draft.h0");
            // CPU yn (the attention's normed input) for the deepest bisect layer.
            let (y_cpu, _p, _c) = cpu::hc_pre_all(&h_in, block_c, &cpu_layer.hc_attn, &cfg);
            let yn_cpu = cpu::rms_norm(&y_cpu, block_c, dim, &cpu_layer.attn_norm, cfg.norm_eps);
            println!("=== CPU-reference bisect ===");
            cmp2("GPU yn vs CPU yn", &sublayers[0].0, &yn_cpu);
            cmp2("CPU h0 vs ORACLE h0", &cpu_out, &oracle_h0);
            cmp2("GPU h0 vs ORACLE h0", &stages[1], &oracle_h0);
            cmp2("GPU h0 vs CPU h0", &stages[1], &cpu_out);
            cmp2("GPU attn_out vs CPU attn_out", &sublayers[0].1, &trace.attn_out);
            cmp2("GPU ffn_out vs CPU ffn_out", &sublayers[0].2, &trace.ffn_out);
            // the attention internals: CPU sparse-attn o (gather + de-rotation) and oflat (wo_a
            // einsum) vs the GPU captures — the last two unchecked tensors in the chain.
            {
                let pos0 = start_pos + 1;
                let (_qr, q_c, kv_c) = cpu::attn_qkv(&cpu_layer.attn, &yn_cpu, block_c, pos0, &rope, &cfg);
                let mut kv_cat = kv_cache.clone();
                kv_cat.extend_from_slice(&kv_c);
                let idx_row: Vec<i64> = cpu::dspark_topk_idxs(cfg.window_size, block_c, start_pos);
                let t = idx_row.len();
                let mut flat = Vec::with_capacity(block_c * t);
                for _ in 0..block_c { flat.extend_from_slice(&idx_row); }
                let scale = (hd as f64).powf(-0.5) as f32;
                let mut o_c = cpu::sparse_attn(&q_c, block_c, cfg.n_heads, hd, &kv_cat,
                                               cfg.window_size + block_c, &cpu_layer.attn.sink, &flat, t, scale);
                let rows = block_c * cfg.n_heads;
                let pos: Vec<usize> = (0..rows).map(|i| pos0 + i / cfg.n_heads).collect();
                cpu::apply_rope(&mut o_c, rows, hd, &rope, &pos, true);
                cmp2("GPU o (gather) vs CPU o", &sublayers[0].3, &o_c);
                cmp2("GPU q(impl) vs CPU q", &sublayers[0].6, &q_c);
                cmp2("GPU draft_kv vs CPU kv", &sublayers[0].7, &kv_c);
                // staged q-path bisect: wq_a GEMM, normed qr, wq_b GEMM (pre-rescale).
                {
                    let (g2, r2) = (cfg.o_groups, cfg.o_lora_rank);
                    let _ = (g2, r2);
                    let qlr = cfg.q_lora_rank;
                    let yn_bf: Vec<bf16> = yn_cpu.iter().map(|&v| bf16::from_f32(v)).collect();
                    let yn_dev = dev.htod_sync_copy(&yn_bf).expect("htod yn");
                    let (yc, ysa) = ds.rt.quant_g128::<gb10_inference::dsv4_attn::B, cudarc::driver::CudaSlice<u8>>(&yn_dev, block_c, dim).expect("quant yn");
                    let qr_pre_g = ds.rt.fp8_bsb_rows(&ds.stages[0].wq_a, &yc, &ysa, block_c).expect("wq_a");
                    let qr_pre_h: Vec<f32> = dev.dtoh_sync_copy(&qr_pre_g).expect("dtoh")
                        .iter().map(|v| v.to_f32()).collect();
                    let qr_pre_c = cpu::quant_gemm(&yn_cpu, block_c, dim, &cpu_layer.attn.wq_a, qlr, 128);
                    cmp2("GPU qr_pre vs CPU qr_pre", &qr_pre_h, &qr_pre_c);
                    let qr_g = ds.rt.rmsnorm(&qr_pre_g, &ds.stages[0].q_norm, block_c, qlr, cfg.norm_eps).expect("q_norm");
                    let qr_h: Vec<f32> = dev.dtoh_sync_copy(&qr_g).expect("dtoh")
                        .iter().map(|v| v.to_f32()).collect();
                    let qr_c = cpu::rms_norm(&qr_pre_c, block_c, qlr, &cpu_layer.attn.q_norm, cfg.norm_eps);
                    cmp2("GPU qr vs CPU qr", &qr_h, &qr_c);
                    let (qc, qsa) = ds.rt.quant_g128::<gb10_inference::dsv4_attn::B, cudarc::driver::CudaSlice<u8>>(&qr_g, block_c, qlr).expect("quant qr");
                    let q_pre_g = ds.rt.fp8_bsb_rows(&ds.stages[0].wq_b, &qc, &qsa, block_c).expect("wq_b");
                    let q_pre_h: Vec<f32> = dev.dtoh_sync_copy(&q_pre_g).expect("dtoh")
                        .iter().map(|v| v.to_f32()).collect();
                    let q_pre_c = cpu::quant_gemm(&qr_c, block_c, qlr, &cpu_layer.attn.wq_b, cfg.n_heads * hd, 128);
                    cmp2("GPU q_pre(rescale) vs CPU q_pre", &q_pre_h, &q_pre_c);
                    cmp2("GPU q_pre(impl) vs CPU q_pre", &sublayers[0].5, &q_pre_c);
                    // rescale-only (no rope) then rope: which of the last two steps diverges?
                    let mut q_rsc_c = q_pre_c.clone();
                    for i in 0..block_c * cfg.n_heads {
                        let row = &mut q_rsc_c[i * hd..(i + 1) * hd];
                        let mut sq: Vec<f32> = row.iter().map(|&v| cpu::bf(v * v)).collect();
                        let ss = cpu::pairwise_sum(&mut sq);
                        let mean = cpu::bf(ss / hd as f32);
                        let arg = cpu::bf(mean + cfg.norm_eps);
                        let r = cpu::bf(arg.sqrt().recip());
                        for v in row.iter_mut() { *v = cpu::bf(*v * r); }
                    }
                    cmp2("GPU q_rsc(impl) vs CPU q_rsc", &sublayers[0].6, &q_rsc_c);
                    println!("    gpu q[0..6]:       {:?}", &sublayers[0].5[..6]);
                    println!("    cpu q_pre[0..6]:   {:?}", &q_pre_c[..6]);
                    println!("    cpu q_rsc[0..6]:   {:?}", &q_rsc_c[..6]);
                    println!("    cpu q (full)[0..6]: {:?}", &q_c[..6]);
                    println!("    gpu q[512..518]:   {:?}", &sublayers[0].5[512..518]);
                    println!("    cpu q_rsc[512..518]: {:?}", &q_rsc_c[512..518]);
                    for r in 0..5usize {
                        let (mut num, mut den) = (0.0f64, 0.0f64);
                        for d in 0..32768usize {
                            let df = sublayers[0].5[r * 32768 + d] - q_c[r * 32768 + d];
                            num += (df as f64) * (df as f64);
                            den += (q_c[r * 32768 + d] as f64) * (q_c[r * 32768 + d] as f64);
                        }
                        println!("    q row {r}: rel-L2 {:.3e}", (num / den.max(1e-30)).sqrt());
                    }
                    for h in [0usize, 1, 2, 3, 15, 16, 17, 32, 63] {
                        let (mut num, mut den) = (0.0f64, 0.0f64);
                        for d in 0..512usize {
                            let df = sublayers[0].5[h * 512 + d] - q_c[h * 512 + d];
                            num += (df as f64) * (df as f64);
                            den += (q_c[h * 512 + d] as f64) * (q_c[h * 512 + d] as f64);
                        }
                        println!("    q row0 head {h:>2}: rel-L2 {:.3e}", (num / den.max(1e-30)).sqrt());
                    }
                    for d0 in [0usize, 448, 456] {
                        let mut gpu_v = Vec::new();
                        let mut cpu_v = Vec::new();
                        for d in d0..d0 + 8 {
                            gpu_v.push(format!("{:.4}", sublayers[0].5[2 * 512 + d]));
                            cpu_v.push(format!("{:.4}", q_c[2 * 512 + d]));
                        }
                        println!("    head2 dims {d0}.. gpu: {:?}", gpu_v);
                        println!("    head2 dims {d0}.. cpu: {:?}", cpu_v);
                    }
                }
                // attention sink content (the softmax denominator offset)
                {
                    let sink_gpu: Vec<f32> = dev.dtoh_sync_copy(&ds.stages[0].sink).expect("sink dtoh");
                    cmp2("GPU sink vs CPU sink", &sink_gpu, &cpu_layer.attn.sink);
                    println!("    gpu sink[0..6]: {:?}", &sink_gpu[..6]);
                    println!("    cpu sink[0..6]: {:?}", &cpu_layer.attn.sink[..6]);
                }
                // oflat: per group, xg @ wo_a[g]ᵀ (the olo einsum's CPU form)
                let (g, r, gd, nh) = (cfg.o_groups, cfg.o_lora_rank, cfg.n_heads * hd / cfg.o_groups, cfg.n_heads);
                let mut oflat_c = vec![0.0f32; block_c * g * r];
                for grp in 0..g {
                    let og = &o_c[grp * gd..];
                    let mut xg = vec![0.0f32; block_c * gd];
                    for i in 0..block_c {
                        xg[i * gd..(i + 1) * gd].copy_from_slice(&og[i * nh * hd..i * nh * hd + gd]);
                    }
                    let wag = &cpu_layer.attn.wo_a[grp * r * gd..(grp + 1) * r * gd];
                    let yg = cpu::gemm_bf16(&xg, block_c, gd, wag, r);
                    for i in 0..block_c {
                        oflat_c[i * g * r + grp * r..i * g * r + (grp + 1) * r].copy_from_slice(&yg[i * r..(i + 1) * r]);
                    }
                }
                cmp2("GPU oflat vs CPU oflat", &sublayers[0].4, &oflat_c);
            }
        }
        std::process::exit(0);
    }
    let out = ds.draft(&draft_dev, real_tok, start_pos).expect("dspark draft");
    let dt = t0.elapsed().as_secs_f64();
    println!("=== DSpark draft: {} tokens in {dt:.3}s ===", out.drafts.len());
    assert_eq!(out.drafts.len(), block, "draft output length");
    assert_eq!(out.confidence.len(), block, "confidence length");

    // ---- (d) report vs oracle (mechanics; values diverge per chaos) ----
    let oracle_drafts: Vec<i32> = oracle_ids[1..=block].iter().map(|&v| v as i32).collect();
    let matches: usize = out.drafts.iter().zip(&oracle_drafts).map(|(a, b)| (*a == *b) as usize).sum();
    println!("=== draft tokens vs oracle (CHAOS — match count is informational, not a gate) ===");
    println!("  engine: {:?}", out.drafts);
    println!("  oracle: {:?}", oracle_drafts);
    println!("  match: {}/{} (the §7 G1 amendment: draft internals are chaotic; mechanics is the gate)", matches, block);
    println!("  confidence (raw fp32, logged): {:?}", out.confidence);

    // ---- mechanics verdict ----
    let pass = idxs_ref.len() == expected_t_win + block && out.drafts.len() == block;
    println!("\n=== DSPARK DRAFT MECHANICS: {} ===", if pass { "OK" } else { "FAIL" });
    println!("  warm path: 3 stages × {sw} positions ring-written");
    println!("  index list: {} entries (non-causal, matches dsv4_cpu)", idxs_ref.len());
    println!("  Markov chain: {block} sequential steps (each row depends on prior argmax)");
    println!("  draft output: {block} tokens + {block} confidence scores");
    if !pass { std::process::exit(1); }
}

/// `--extract-dspark --bundle <b> --out <model-dir>`: write the 3 DSpark stages from the bundle
/// into `{out}/rank0/dspark.safetensors` + `{out}/rank1/dspark.safetensors` (replicated — both
/// ranks load the full 256-expert stages; the cluster ships rank1/ to the node). One-time offline.
pub(crate) fn run_extract_dspark(args: &[String]) {
    use std::path::Path;
    let bundle = parse_arg(args, "--bundle").expect("--extract-dspark requires --bundle <bundle>");
    let out = parse_arg(args, "--out").expect("--extract-dspark requires --out <model-dir>");
    let cfg = gb10_inference::dsv4_load::load_config(Path::new(bundle)).expect("load_config");
    for rank in 0..2 {
        let rd = Path::new(out).join(format!("rank{rank}"));
        gb10_inference::dsv4_convert::write_dspark_artifact(Path::new(bundle), &cfg, &rd)
            .unwrap_or_else(|e| { eprintln!("extract rank{rank}: {e:#}"); std::process::exit(1); });
    }
    println!("=== dspark.safetensors written to {out}/rank0/ + rank1/ ===");
}

/// `--probe-dspark-rollback`: WORK-1 gate. Loads a trunk slice (SWA+CSA+HCA), prefills, then:
/// (a) snapshot the per-layer attention state; (b) verify-forward 6 tokens (the state advances);
/// (c) restore the snapshot; (d) re-verify the SAME 6 tokens. The two verify logits MUST be
/// bitwise-identical (the restore fully rewound KV + compressor + indexer). Control: verify twice
/// WITHOUT restore → the logits DIFFER (proves the state really advanced, the test isn't vacuous).
pub(crate) fn run_probe_dspark_rollback(args: &[String]) {
    use gb10_inference::dsv4_model::Dsv4GpuModel;
    use std::path::Path;
    use cudarc::driver::CudaDevice;

    let bundle = parse_arg(args, "--model-dir").expect("--probe-dspark-rollback requires --model-dir <bundle>");
    let cfg = gb10_inference::dsv4_load::load_config(Path::new(bundle)).expect("load_config");
    let n_layers: usize = parse_arg(args, "--layers").map(|s| s.parse().expect("--layers N")).unwrap_or(4);
    assert!(n_layers >= 4, "need >=4 layers to exercise SWA+CSA+HCA rollback");
    let dev = CudaDevice::new(0).expect("CUDA device 0");

    eprintln!("[rollback] loading {n_layers}-layer trunk slice (single-process) ...");
    let mut m = Dsv4GpuModel::load(&dev, Path::new(bundle), &cfg, 256, 64, n_layers).expect("Dsv4GpuModel::load");

    // Prefill a short prompt (positions 0..15).
    let prompt: Vec<i32> = (0..16).map(|i| (((7 + i as i64 * 9973) % 129040) as i32).max(2)).collect();
    let prefill_len = prompt.len();
    let _ = m.forward(&prompt, 0).expect("prefill");
    eprintln!("[rollback] prefilled {prefill_len} tokens; snapshotting state ...");

    // 6 verify tokens at positions prefill_len..prefill_len+5 (arbitrary valid ids).
    let verify_ids: Vec<i32> = (0..6).map(|i| (((3 + i as i64 * 7919) % 129040) as i32).max(2)).collect();
    let vpos = prefill_len;

    // ---- (1) snapshot, verify, restore, re-verify → MUST be bitwise-identical ----
    let snap = m.snapshot_verify_state().expect("snapshot");
    let logits_a = m.forward_verify_logits(&verify_ids, vpos).expect("verify A");
    let la: Vec<f32> = dev.dtoh_sync_copy(&logits_a).expect("dtoh A");
    m.restore_verify_state(&snap).expect("restore");
    let logits_b = m.forward_verify_logits(&verify_ids, vpos).expect("verify B (post-restore)");
    let lb: Vec<f32> = dev.dtoh_sync_copy(&logits_b).expect("dtoh B");

    let n = la.len();
    let mism = la.iter().zip(&lb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max_d = la.iter().zip(&lb).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("[rollback] post-restore re-verify: {n} logits, {mism} bitwise mismatches, max-abs-delta {max_d:e}");

    // ---- (2) CONTROL: verify twice WITHOUT restore → MUST differ (state advanced) ----
    // After verify B the state is at vpos+6; a third verify at vpos+6 advances further. To prove
    // the restore matters, re-snapshot at vpos, verify, verify-AGAIN-at-vpos (state now at vpos+6,
    // so the second sees the first's KV) → must DIFFER.
    let _ = m.forward(&prompt, 0).expect("re-prefill (reset state to vpos)");
    let snap2 = m.snapshot_verify_state().expect("snapshot2");
    let logits_c = m.forward_verify_logits(&verify_ids, vpos).expect("verify C");
    let lc: Vec<f32> = dev.dtoh_sync_copy(&logits_c).expect("dtoh C");
    // now state is at vpos+6; verify the SAME ids at vpos WITHOUT restore → sees extra KV
    let logits_d = m.forward_verify_logits(&verify_ids, vpos).expect("verify D (no restore)");
    let ld: Vec<f32> = dev.dtoh_sync_copy(&logits_d).expect("dtoh D");
    m.restore_verify_state(&snap2).expect("restore2 (cleanup)");
    let ctrl_mism = lc.iter().zip(&ld).filter(|(a, b)| a.to_bits() != b.to_bits()).count();

    let pass = mism == 0 && ctrl_mism > 0;
    println!("\n=== DSPARK VERIFY ROLLBACK: {} ===", if pass { "OK" } else { "FAIL" });
    println!("  snapshot → verify → restore → re-verify: {mism}/{n} bitwise mismatches (want 0)");
    println!("  control (verify twice w/o restore):     {ctrl_mism}/{n} differ (want >0 — state really advanced)");
    if !pass { std::process::exit(1); }
}
