//! DSpark fp8 draft-logits gate (Tier 1.1b, --dspark-fp8-logits):
//!   1. GRAPH ARM: the graphed fp8 draft must equal the eager fp8 draft BITWISE (same
//!      kernels/args — the bf16 readout path included).
//!   2. QUALITY: the fp8 draft logits must stay close to the bf16-head logits (rel-L2) and
//!      keep near-total per-row argmax agreement — fp8 noise may flip near-ties (acceptance
//!      cost) but must not shift the distribution. Asserts the SIGNAL (rel-L2 + agreement
//!      rate), never the absence of errors.
//! A/B rides `use_fp8_logits` (per-call arm select built for exactly this).

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::CudaDevice;
use half::bf16;

use gb10_inference::{dsv4_dspark::Dsv4DSpark, dsv4_load, dsv4_model::Dsv4GpuModel};

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";

fn gate() -> MutexGuard<'static, ()> {
    static G: Mutex<()> = Mutex::new(());
    G.lock().unwrap_or_else(|e| e.into_inner())
}

fn synth_mh(rows: usize, three_d: usize, seed: u64) -> Vec<bf16> {
    (0..rows * three_d)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2654435761).wrapping_add(seed) % 2000;
            bf16::from_f32((h as f32 / 2000.0 - 0.5) * 0.05)
        })
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}

#[test]
fn dspark_fp8_logits_quality_and_graph_bitwise() {
    let _g = gate();
    // CLI-1: the options registry (no env); set before any GPU work.
    gb10_inference::opts::test_set(gb10_inference::opt!("dspark-fp8-logits"), Some("1"));
    gb10_inference::opts::test_set(gb10_inference::opt!("dspark-graph"), Some("1"));
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).unwrap();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let three_d = 3 * cfg.dim;
    let sw = 130usize;
    let vocab = cfg.vocab_size;
    let block = cfg.dspark_block_size;

    let top = Dsv4GpuModel::load_trunk_top(&dev, bundle, &cfg).expect("load_trunk_top");
    let embed = top.embed.clone();
    let head = top.head.clone();
    drop(top);
    let mut ds = Dsv4DSpark::load(&dev, bundle, &cfg, 2048, embed, head).expect("Dsv4DSpark::load");
    assert!(ds.use_fp8_logits, "fp8 arm should be armed by the env flag");

    let warm = dev.htod_sync_copy(&synth_mh(sw, three_d, 7)).expect("htod warm");
    ds.warm(&warm, sw).expect("warm");
    let draft_mh = dev.htod_sync_copy(&synth_mh(1, three_d, 99)).expect("htod draft mh");

    let mut steps: Vec<(usize, i32)> = (0..8).map(|i| (130 + i * 6, 12345 + i as i32 * 31)).collect();
    steps.push((60, 777));
    steps.push((90, 888));

    let (mut num, mut den) = (0.0f64, 0.0f64);
    let (mut rows_arg_same, mut rows_total) = (0usize, 0usize);
    let (mut ids_same, mut ids_total) = (0usize, 0usize);
    for (step, (sp, tok)) in steps.iter().copied().enumerate() {
        // bf16-head eager reference.
        ds.use_fp8_logits = false;
        let (ref_out, ref_logits) = ds.draft_full(&draft_mh, tok, sp).expect("eager bf16 draft");
        // fp8 eager + fp8 graphed.
        ds.use_fp8_logits = true;
        let (f_out, f_logits) = ds.draft_full(&draft_mh, tok, sp).expect("eager fp8 draft");
        let (g_out, g_logits) = ds.draft_graphed_full(&draft_mh, tok, sp).expect("graphed fp8 draft");
        // 1. graph ≡ eager fp8 BITWISE.
        let mism = f_logits.iter().zip(g_logits.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(mism, 0, "graphed != eager fp8 logits at sp={sp} (step {step})");
        assert_eq!(f_out.drafts, g_out.drafts, "graphed != eager fp8 draft ids at sp={sp}");
        // 2. fp8 vs bf16-head quality.
        for i in 0..ref_logits.len() {
            let d = f_logits[i] - ref_logits[i];
            num += (d as f64) * (d as f64);
            den += (ref_logits[i] as f64) * (ref_logits[i] as f64);
        }
        for r in 0..block {
            rows_total += 1;
            if argmax(&f_logits[r * vocab..(r + 1) * vocab]) == argmax(&ref_logits[r * vocab..(r + 1) * vocab]) {
                rows_arg_same += 1;
            }
        }
        ids_total += block;
        ids_same += f_out.drafts.iter().zip(ref_out.drafts.iter()).filter(|(a, b)| a == b).count();
        eprintln!(
            "[fp8-gate] step {step} sp={sp}: graph-bitwise 0/{} ids_fp8={:?} ids_bf16={:?}",
            f_logits.len(), f_out.drafts, ref_out.drafts
        );
    }
    let rel_l2 = (num / den.max(1e-30)).sqrt();
    let arg_agree = rows_arg_same as f64 / rows_total as f64;
    let id_agree = ids_same as f64 / ids_total as f64;
    println!(
        "DSPARK-FP8-GATE: rel-L2(fp8 vs bf16 head) {rel_l2:.3e}; row-argmax agreement {:.1}% ({rows_arg_same}/{rows_total}); draft-id agreement {:.1}% ({ids_same}/{ids_total}); graph≡eager-fp8 bitwise over {} steps",
        100.0 * arg_agree, 100.0 * id_agree, steps.len()
    );
    // Measured on this SYNTHETIC input (2026-07-30): rel-L2 2.8e-2 (e4m3 weights + e4m3
    // activations — in-family), argmax agreement 92%, id agreement 84%. Synthetic random
    // hiddens exaggerate near-ties vs real activations; the production acceptance gate is
    // the TP serve (acceptance + LOSSLESS ids), not these numbers. The asserts below keep
    // this a SIGNAL test (catch a quantizer-layout break = rel-L2 ~1e0, agreement ~0%).
    assert!(rel_l2 < 6e-2, "fp8 logits drifted: rel-L2 {rel_l2:.3e}");
    assert!(arg_agree >= 0.80, "row-argmax agreement too low: {:.1}%", 100.0 * arg_agree);
    assert!(id_agree >= 0.70, "draft-id agreement too low: {:.1}%", 100.0 * id_agree);
    println!("DSPARK-FP8-GATE: PASS");
}
