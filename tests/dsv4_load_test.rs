//! Lane A loader gate tests (G1). Run:
//!   cargo test --release --test dsv4_load_test -- --nocapture
//!
//! The heavyweight tests stream real checkpoint shards from
//! /mnt/models/DeepSeek-V4-Flash-DSpark (READ-ONLY) and serialize on a static
//! gate so the ~200 GB of disk traffic doesn't thrash concurrently.

use gb10_inference::dsv4_load::*;
use gb10_inference::quant;
use half::bf16;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";

fn bundle_present() -> bool { Path::new(BUNDLE).join("inference/config.json").exists() }

// 0731-native fixture (regenerated 2026-08-01 from the DeepSeek-V4-Flash-0731 bundle).
// SKIP when absent (NOT a hard panic) — the gate is a cross-check against convert.py's dequant,
// not the load-bearing contract; an absent fixture means "regenerate, don't fail the suite".
const WO_A_REF: &str = "/tmp/wo_a_ref.npy"; // regenerate: see wo_a_dequant_bitwise_vs_oracle

/// Serialize the three shard-streaming tests (strict-load, A/B, wo_a).
static GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn bundle() -> PathBuf { PathBuf::from(BUNDLE) }

fn assert_f32(t: &HostTensor, shape: &[usize], key: &str) {
    match t {
        HostTensor::F32 { shape: s, .. } => assert_eq!(s, shape, "{key} shape"),
        other => panic!("{key}: expected F32 {shape:?}, got {other:?}"),
    }
}

fn assert_bf16(t: &HostTensor, shape: &[usize], key: &str) {
    match t {
        HostTensor::BF16 { shape: s, .. } => assert_eq!(s, shape, "{key} shape"),
        other => panic!("{key}: expected BF16 {shape:?}, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. Config parse — §D real values, never the ModelArgs demo defaults.
// ---------------------------------------------------------------------------
#[test]
fn config_real_values() {
    if !bundle_present() { return; } // lab-model-bundle gate: skipped where the DSV4 bundle is absent (dev box)
    let cfg = load_config(&bundle()).expect("load_config");

    assert_eq!(cfg.vocab_size, 129280);
    assert_eq!(cfg.dim, 4096);
    assert_eq!(cfg.moe_inter_dim, 2048);
    assert_eq!(cfg.n_layers, 43, "§D: 43 trunk layers (ModelArgs demo says 7)");
    assert_eq!(cfg.n_hash_layers, 3);
    assert_eq!(cfg.n_mtp_layers, 3, "§D: real is 3 (HF num_nextn_predict_layers:1 is a lie)");
    assert_eq!(cfg.dspark_block_size, 5, "§D: 5 (ModelArgs demo 0 disables MTP)");
    assert_eq!(cfg.dspark_noise_token_id, 128799);
    assert_eq!(cfg.dspark_target_layer_ids, vec![40, 41, 42]);
    assert_eq!(cfg.dspark_markov_rank, 256);
    assert_eq!(cfg.n_heads, 64);
    assert_eq!(cfg.n_routed_experts, 256, "§D: 256 experts (demo says 8)");
    assert_eq!(cfg.n_shared_experts, 1);
    assert_eq!(cfg.n_activated_experts, 6, "§D: top-6 (demo says 2)");
    assert_eq!(cfg.route_scale, 1.5);
    assert_eq!(cfg.swiglu_limit, 10.0, "§D: 10 (demo 0 disables the clamps)");
    assert_eq!(cfg.q_lora_rank, 1024);
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.rope_head_dim, 64);
    assert_eq!(cfg.o_groups, 8);
    assert_eq!(cfg.o_lora_rank, 1024);
    assert_eq!(cfg.window_size, 128);
    assert_eq!(cfg.original_seq_len, 65536);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.rope_factor, 16.0, "§D: 16 (demo says 40)");
    assert_eq!(cfg.beta_fast, 32);
    assert_eq!(cfg.beta_slow, 1);
    assert_eq!(cfg.index_n_heads, 64);
    assert_eq!(cfg.index_head_dim, 128);
    assert_eq!(cfg.index_topk, 512);
    assert_eq!(cfg.hc_mult, 4);
    assert_eq!(cfg.hc_sinkhorn_iters, 20);
    assert_eq!(cfg.compress_rope_theta, 160000.0, "§D: 160000 (demo says 40000)");
    // eps values are NOT in the json — reference-hardcoded (model.py:62, :81).
    assert_eq!(cfg.norm_eps, 1e-6);
    assert_eq!(cfg.hc_eps, 1e-6);

    // 46-entry compress_ratios: 0,0 then alternating 4/128 for 2..=42, then 0,0,0.
    assert_eq!(cfg.compress_ratios.len(), 46, "§D: 46 entries (demo has 8)");
    assert_eq!(cfg.compress_ratios[0], 0);
    assert_eq!(cfg.compress_ratios[1], 0);
    for i in 2..=42usize {
        let want = if i % 2 == 0 { 4 } else { 128 };
        assert_eq!(cfg.compress_ratios[i], want, "compress_ratios[{i}]");
    }
    assert_eq!(&cfg.compress_ratios[43..46], &[0, 0, 0]);

    // Layer-kind map (§A.2): 0,1 SWA; even 2..42 CSA (21); odd 3..41 HCA (20);
    // 43..45 DSpark (SWA).
    assert_eq!(cfg.layer_kind(0), LayerKind::Swa);
    assert_eq!(cfg.layer_kind(1), LayerKind::Swa);
    assert_eq!(cfg.layer_kind(2), LayerKind::Csa);
    assert_eq!(cfg.layer_kind(3), LayerKind::Hca);
    assert_eq!(cfg.layer_kind(4), LayerKind::Csa);
    assert_eq!(cfg.layer_kind(41), LayerKind::Hca);
    assert_eq!(cfg.layer_kind(42), LayerKind::Csa);
    assert_eq!(cfg.layer_kind(43), LayerKind::Swa);
    assert_eq!(cfg.layer_kind(45), LayerKind::Swa);
    assert!(cfg.is_hash_layer(0) && cfg.is_hash_layer(2));
    assert!(!cfg.is_hash_layer(3) && !cfg.is_hash_layer(42));

    // §D "max_seq_len=1048576 handling": inference/config.json carries NO max_seq_len
    // (ModelArgs' demo default 4096 must never leak in; the serving 1M value lives in
    // the HF config's max_position_embeddings). Assert the trap stays documented.
    let raw = std::fs::read_to_string(Path::new(BUNDLE).join("inference/config.json")).unwrap();
    assert!(!raw.contains("max_seq_len"),
            "inference/config.json grew a max_seq_len — revisit the §D trap handling");

    println!("config_real_values: all §D values OK");
}

// ---------------------------------------------------------------------------
// 4. npy round-trip (f32 + i64), v1.0 little-endian.
// ---------------------------------------------------------------------------
#[test]
fn npy_roundtrip() {
    let dir = std::env::temp_dir();

    let f32_path = dir.join("dsv4_npy_rt_f32.npy");
    let f32_data = [0.0f32, -0.0, 1.5, -2.25, 1e-30, 3.4e38, f32::MIN_POSITIVE, -6.0];
    let f32_shape = [2usize, 4];
    write_npy_f32(&f32_path, &f32_shape, &f32_data).unwrap();
    let (shape, data) = read_npy(&f32_path).unwrap();
    assert_eq!(shape, f32_shape);
    match data {
        NpyData::F32(v) => {
            assert_eq!(v.len(), f32_data.len());
            for (a, b) in v.iter().zip(f32_data.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "f32 round-trip bits");
            }
        }
        other => panic!("expected F32, got {other:?}"),
    }

    // 1-D shape exercises the "(n,)" tuple form.
    let i64_path = dir.join("dsv4_npy_rt_i64.npy");
    let i64_data = [0i64, -1, 1, i64::MIN + 1, i64::MAX];
    write_npy_i64(&i64_path, &[5], &i64_data).unwrap();
    let (shape, data) = read_npy(&i64_path).unwrap();
    assert_eq!(shape, vec![5]);
    match data {
        NpyData::I64(v) => assert_eq!(v, i64_data),
        other => panic!("expected I64, got {other:?}"),
    }

    println!("npy_roundtrip: f32 {} bytes + i64 {} bytes OK",
             std::fs::metadata(&f32_path).unwrap().len(),
             std::fs::metadata(&i64_path).unwrap().len());
}

// ---------------------------------------------------------------------------
// 1. STRICT-LOAD GATE: all 43 trunk layers + 3 mtp stages, zero missing AND zero
//    unexpected keys. Streams the full bundle once.
// ---------------------------------------------------------------------------
#[test]
fn strict_load_all_46_moe_blocks() {
    if !bundle_present() { return; } // lab-model-bundle gate: skipped where the DSV4 bundle is absent (dev box)
    let _g = gate();
    let cfg = load_config(&bundle()).unwrap();
    let bundle = bundle();

    // Global coverage: every checkpoint key belongs to exactly one strict-loaded
    // block or the trunk top (72317 total = 72311 block keys + 6 top-level).
    let wm = load_weight_map(&bundle).unwrap();
    let block_keys = wm.keys().filter(|k| k.starts_with("layers.") || k.starts_with("mtp.")).count();
    assert_eq!(wm.len(), 72317, "checkpoint census (§F.1)");
    assert_eq!(block_keys, 72311, "all non-top keys are covered by the 46 strict loads");

    let check_common = |l: &Dsv4Layer, tag: &str| {
        assert_eq!(l.experts_w1.len(), 256, "{tag} experts_w1");
        assert_eq!(l.experts_w2.len(), 256, "{tag} experts_w2");
        assert_eq!(l.experts_w3.len(), 256, "{tag} experts_w3");
        for (e, t) in l.experts_w1.iter().enumerate().step_by(255) {
            assert_eq!((t.m, t.k), (2048, 4096), "{tag} w1 expert {e}");
        }
        for (e, t) in l.experts_w2.iter().enumerate().step_by(255) {
            assert_eq!((t.m, t.k), (4096, 2048), "{tag} w2 expert {e}");
        }
        for (e, t) in l.experts_w3.iter().enumerate().step_by(255) {
            assert_eq!((t.m, t.k), (2048, 4096), "{tag} w3 expert {e}");
        }
        let t = &l.tensors;
        assert_bf16(&t["attn.wo_a.weight"], &[8192, 4096], "wo_a (§F.2)");
        assert_f32(&t["attn.wq_a.weight"], &[1024, 4096], "wq_a");
        assert_f32(&t["attn.wq_b.weight"], &[32768, 1024], "wq_b");
        assert_f32(&t["attn.wkv.weight"], &[512, 4096], "wkv");
        assert_f32(&t["attn.wo_b.weight"], &[4096, 8192], "wo_b");
        assert_f32(&t["attn.q_norm.weight"], &[1024], "q_norm");
        assert_f32(&t["attn.kv_norm.weight"], &[512], "kv_norm");
        assert_f32(&t["attn.attn_sink"], &[64], "attn_sink");
        assert_f32(&t["attn_norm.weight"], &[4096], "attn_norm");
        assert_f32(&t["ffn_norm.weight"], &[4096], "ffn_norm");
        assert_f32(&t["hc_attn_fn"], &[24, 16384], "hc_attn_fn");
        assert_f32(&t["hc_attn_base"], &[24], "hc_attn_base");
        assert_f32(&t["hc_attn_scale"], &[3], "hc_attn_scale");
        assert_f32(&t["hc_ffn_fn"], &[24, 16384], "hc_ffn_fn");
        assert_bf16(&t["ffn.gate.weight"], &[256, 4096], "ffn.gate.weight");
        assert_f32(&t["ffn.shared_experts.w1.weight"], &[2048, 4096], "shared w1");
        assert_f32(&t["ffn.shared_experts.w2.weight"], &[4096, 2048], "shared w2");
        assert_f32(&t["ffn.shared_experts.w3.weight"], &[2048, 4096], "shared w3");
        // wo_a.scale and every fp8 .scale must be consumed, never surfaced.
        for k in t.keys() {
            assert!(!k.ends_with(".scale"), "{tag}: unconsumed scale key {k}");
        }
    };

    let mut total_loaded = 0usize;
    for layer in 0..cfg.n_layers {
        let tag = format!("layers.{layer}");
        let l = load_layer(&bundle, &cfg, layer)
            .unwrap_or_else(|e| panic!("{tag}: strict load failed: {e:#}"));
        check_common(&l, &tag);
        let kind = cfg.layer_kind(layer);
        let hash = cfg.is_hash_layer(layer);
        // Routing side: tid2eid on 0..2 (I32!), gate.bias on >=3.
        if hash {
            match &l.tensors["ffn.gate.tid2eid"] {
                HostTensor::I32 { shape, data } => {
                    assert_eq!(shape, &[129280, 6], "tid2eid shape");
                    assert!(data.iter().all(|&v| (0..256).contains(&v)), "tid2eid values are expert ids");
                }
                other => panic!("tid2eid: expected I32, got {other:?}"),
            }
            assert!(!l.tensors.contains_key("ffn.gate.bias"), "{tag}: hash layer must not have gate.bias");
        } else {
            assert_f32(&l.tensors["ffn.gate.bias"], &[256], "ffn.gate.bias");
            assert!(!l.tensors.contains_key("ffn.gate.tid2eid"), "{tag}: non-hash layer must not have tid2eid");
        }
        // Attention-variant tensors.
        match kind {
            LayerKind::Swa => {
                assert!(!l.tensors.keys().any(|k| k.contains("compressor") || k.contains("indexer")),
                        "{tag}: SWA layer must not own compressor/indexer tensors");
            }
            LayerKind::Csa => {
                assert_f32(&l.tensors["attn.compressor.wkv.weight"], &[1024, 4096], "csa wkv");
                assert_f32(&l.tensors["attn.compressor.wgate.weight"], &[1024, 4096], "csa wgate");
                assert_f32(&l.tensors["attn.compressor.norm.weight"], &[512], "csa cnorm");
                assert_f32(&l.tensors["attn.compressor.ape"], &[4, 1024], "csa ape");
                assert_f32(&l.tensors["attn.indexer.wq_b.weight"], &[8192, 1024], "idx wq_b");
                assert_bf16(&l.tensors["attn.indexer.weights_proj.weight"], &[64, 4096], "idx wproj");
                assert_f32(&l.tensors["attn.indexer.compressor.wkv.weight"], &[256, 4096], "idx c wkv");
                assert_f32(&l.tensors["attn.indexer.compressor.wgate.weight"], &[256, 4096], "idx c wgate");
                assert_f32(&l.tensors["attn.indexer.compressor.norm.weight"], &[128], "idx c norm");
                assert_f32(&l.tensors["attn.indexer.compressor.ape"], &[4, 256], "idx c ape");
            }
            LayerKind::Hca => {
                assert_f32(&l.tensors["attn.compressor.wkv.weight"], &[512, 4096], "hca wkv");
                assert_f32(&l.tensors["attn.compressor.wgate.weight"], &[512, 4096], "hca wgate");
                assert_f32(&l.tensors["attn.compressor.norm.weight"], &[512], "hca cnorm");
                assert_f32(&l.tensors["attn.compressor.ape"], &[128, 512], "hca ape");
                assert!(!l.tensors.keys().any(|k| k.contains("indexer")), "{tag}: HCA has no indexer");
            }
        }
        let want_map = match (kind, hash) {
            (LayerKind::Swa, true) => 21,
            (LayerKind::Csa, true) => 31,
            (LayerKind::Csa, false) => 31,
            (LayerKind::Hca, false) => 25,
            other => panic!("unexpected trunk kind/hash {other:?}"),
        };
        assert_eq!(l.tensors.len(), want_map, "{tag} tensor-map size");
        total_loaded += 1;
        println!("  {tag} ({kind:?}{}) : {} tensors + 3x256 experts — strict OK",
                 if hash { ",hash" } else { "" }, l.tensors.len());
    }

    for stage in 0..cfg.n_mtp_layers {
        let tag = format!("mtp.{stage}");
        let l = load_mtp_stage(&bundle, &cfg, stage)
            .unwrap_or_else(|e| panic!("{tag}: strict load failed: {e:#}"));
        check_common(&l, &tag);
        assert_f32(&l.tensors["ffn.gate.bias"], &[256], "mtp gate.bias");
        assert!(!l.tensors.contains_key("ffn.gate.tid2eid"), "{tag}: DSpark never hash-routes");
        assert!(!l.tensors.keys().any(|k| k.contains("compressor") || k.contains("indexer")),
                "{tag}: DSpark block has no compressor/indexer");
        assert!(!l.tensors.keys().any(|k| k.contains("embed")), "{tag}: embed must be skipped (tied)");
        match stage {
            0 => {
                assert_f32(&l.tensors["main_proj.weight"], &[4096, 12288], "main_proj");
                assert_f32(&l.tensors["main_norm.weight"], &[4096], "main_norm");
                assert_eq!(l.tensors.len(), 23, "{tag} tensor-map size");
            }
            1 => assert_eq!(l.tensors.len(), 21, "{tag} tensor-map size"),
            2 => {
                assert_f32(&l.tensors["norm.weight"], &[4096], "mtp.2 norm");
                assert_f32(&l.tensors["hc_head_fn"], &[4, 16384], "mtp.2 hc_head_fn");
                assert_f32(&l.tensors["hc_head_base"], &[4], "mtp.2 hc_head_base");
                assert_f32(&l.tensors["hc_head_scale"], &[1], "mtp.2 hc_head_scale");
                assert_bf16(&l.tensors["markov_head.markov_w1.weight"], &[129280, 256], "markov_w1");
                assert_f32(&l.tensors["markov_head.markov_w2.weight"], &[129280, 256], "markov_w2");
                assert_f32(&l.tensors["confidence_head.proj.weight"], &[1, 4352], "confidence");
                assert_eq!(l.tensors.len(), 28, "{tag} tensor-map size");
            }
            _ => unreachable!(),
        }
        total_loaded += 1;
        println!("  {tag}: {} tensors + 3x256 experts — strict OK", l.tensors.len());
    }
    assert_eq!(total_loaded, 46);
    println!("STRICT-LOAD GATE PASS: loader strict-loads all 46 MoE blocks \
              (43 trunk + 3 DSpark), zero missing, zero unexpected");
}

// ---------------------------------------------------------------------------
// 2. §F.3 A/B: dequant_nvfp4_f32(repack(w,s)) == e8m0_dequant_fp4_exact(w,s)
//    BITWISE, on the 45 sampled expert tensors.
// ---------------------------------------------------------------------------
#[test]
fn expert_repack_ab_bitwise() {
    if !bundle_present() { return; } // lab-model-bundle gate: skipped where the DSV4 bundle is absent (dev box)
    let _g = gate();
    let bundle = bundle();

    let mut names = Vec::new();
    for layer in [0usize, 2, 3, 20, 42] {
        for e in [0usize, 127, 255] {
            for w in 1..=3u8 {
                names.push(format!("layers.{layer}.ffn.experts.{e}.w{w}.weight"));
                names.push(format!("layers.{layer}.ffn.experts.{e}.w{w}.scale"));
            }
        }
    }
    let raw = stream_raw_tensors(&bundle, &names).unwrap();
    assert_eq!(raw.len(), 90);

    let mut checked = 0usize;
    for layer in [0usize, 2, 3, 20, 42] {
        for e in [0usize, 127, 255] {
            for w in 1..=3u8 {
                let wk = format!("layers.{layer}.ffn.experts.{e}.w{w}.weight");
                let sk = format!("layers.{layer}.ffn.experts.{e}.w{w}.scale");
                let wt = &raw[&wk];
                let st = &raw[&sk];
                assert_eq!(wt.dtype, StDtype::I8, "{wk}");
                assert_eq!(st.dtype, StDtype::F8E8M0, "{sk}");
                let (out, k) = (wt.shape[0], wt.shape[1] * 2);
                assert_eq!(st.shape, vec![out, k / FP4_GROUP], "{sk} geometry");

                let t = repack_expert_fp4_to_nvfp4(&wt.data, &st.data, out, k);
                assert_eq!(t.qweight, wt.data, "{wk}: nibbles must be copied unchanged");
                assert_eq!(t.scales.len(), out * (k / 16));
                assert!(t.global_scale.is_finite() && t.global_scale > 0.0);
                // Cross-check against the existing host codec too (different f32→bf16
                // rounding — not the A/B, just a sanity bound on values).
                let as_bf16 = quant::dequantize_nvfp4(&t);
                assert_eq!(as_bf16.len(), out * k);

                let a = dequant_nvfp4_f32(&t);
                let b = e8m0_dequant_fp4_exact(&wt.data, &st.data, out, k);
                assert_eq!(a.len(), b.len());
                let mism = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                assert_eq!(mism, 0, "{wk}: §F.3 A/B bitwise mismatch ({mism} of {})", a.len());
                checked += 1;
            }
        }
        println!("  layer {layer}: experts 0/127/255 x w1/w2/w3 — bitwise equal");
    }
    assert_eq!(checked, 45);
    println!("§F.3 A/B GATE PASS: 45 sampled expert tensors bitwise-identical \
              (dequant_nvfp4_f32(repack) == e8m0_dequant_fp4_exact)");
}

// ---------------------------------------------------------------------------
// 6. Trunk top level (Lane B's head piece): 6 keys, cast rules, exact upcast.
// ---------------------------------------------------------------------------
#[test]
fn trunk_top_load() {
    if !bundle_present() { return; } // lab-model-bundle gate: skipped where the DSV4 bundle is absent (dev box)
    let _g = gate();
    let cfg = load_config(&bundle()).unwrap();
    let top = load_trunk_top(&bundle(), &cfg).unwrap();
    assert_eq!(top.len(), 6);
    assert_bf16(&top["embed.weight"], &[129280, 4096], "embed");
    assert_f32(&top["head.weight"], &[129280, 4096], "head");
    assert_f32(&top["norm.weight"], &[4096], "norm");
    assert_f32(&top["hc_head_fn"], &[4, 16384], "hc_head_fn");
    assert_f32(&top["hc_head_base"], &[4], "hc_head_base");
    assert_f32(&top["hc_head_scale"], &[1], "hc_head_scale");
    // bf16→f32 upcast is exact: every value's low 16 mantissa bits are zero.
    let HostTensor::F32 { data: norm, .. } = &top["norm.weight"] else { unreachable!() };
    assert!(norm.iter().all(|v| v.to_bits() & 0xFFFF == 0), "norm.weight upcast not exact");
    let HostTensor::F32 { data: head, .. } = &top["head.weight"] else { unreachable!() };
    assert!(head.iter().step_by(4099).all(|v| v.to_bits() & 0xFFFF == 0), "head.weight upcast not exact");
    println!("trunk_top_load: 6 keys OK (embed BF16 kept; norm/head bf16→f32 exact)");
}

// ---------------------------------------------------------------------------
// 3. wo_a dequant (§F.2) bitwise vs the Python oracle on layer 0.
//    Reference: python3 - <<'EOF'   (from the repo root)
//      import sys, numpy as np; sys.path.insert(0, "scripts")
//      import dsv4_ref as r
//      wm = r.load_weight_map(r.BUNDLE)
//      t = r.stream_tensors(r.BUNDLE, wm, ["layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale"])
//      w = r.dequant_wo_a(t["layers.0.attn.wo_a.weight"], t["layers.0.attn.wo_a.scale"])
//      np.save("/tmp/wo_a_ref.npy", w.float().numpy())
//    EOF
// ---------------------------------------------------------------------------
#[test]
fn wo_a_dequant_bitwise_vs_oracle() {
    let _g = gate();
    let ref_path = Path::new(WO_A_REF);
    if !ref_path.exists() {
        eprintln!("SKIP wo_a_dequant_bitwise_vs_oracle: {WO_A_REF} not found — regenerate with scripts/dsv4_ref.py dequant_wo_a (test docstring)");
        return;
    }
    let (shape, data) = read_npy(ref_path).unwrap();
    assert_eq!(shape, vec![8192, 4096]);
    let NpyData::F32(refv) = data else { panic!("{WO_A_REF}: expected F32") };

    let cfg = load_config(&bundle()).unwrap();
    let l = load_layer(&bundle(), &cfg, 0).unwrap();
    let HostTensor::BF16 { shape: got_shape, data: got } = &l.tensors["attn.wo_a.weight"] else {
        panic!("attn.wo_a.weight must be BF16 after §F.2 dequant");
    };
    assert_eq!(got_shape, &[8192, 4096]);
    assert_eq!(got.len(), refv.len());
    let mism = got.iter().zip(refv.iter())
        .filter(|(g, r)| g.to_bits() != bf16::from_f32(**r).to_bits())
        .count();
    assert_eq!(mism, 0, "wo_a bf16 mismatch vs oracle ({mism} of {})", got.len());
    println!("§F.2 wo_a GATE PASS: layer 0 wo_a bf16 bitwise-equal to convert.py dequant ({} values)", got.len());
}
