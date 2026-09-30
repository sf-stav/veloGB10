//! DSpark drafter-graph gate (Tier 1.1a, --dspark-graph): the graphed draft must
//! reproduce the EAGER draft BITWISE — same kernels, same args, same state machine; only
//! the launch vehicle changes. One `Dsv4DSpark` instance runs BOTH arms interleaved at the
//! same start_pos: the draft's only state mutation is the main_kv ring write at slot
//! sp%win (deterministic from the inputs, written before the gather reads), so the two
//! arms see identical state and their logits must be f32-BITWISE equal (a param-patch bug
//! — wrong ring-write sp, wrong draft-idx t, wrong gather length — flips them instantly).
//! 24 steps: capture at sp=130 (t saturates at 128+block), replays at growing sp, plus
//! short-context replays (sp=60/90: t_win < win — exercises the DraftTwin/DraftT patches
//! AND the idxs grid-dim patch). Draft ids must also match (the shared Markov tail).

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

/// Deterministic synthetic main_hidden (mechanics gate — no oracle needed).
fn synth_mh(rows: usize, three_d: usize, seed: u64) -> Vec<bf16> {
    (0..rows * three_d)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2654435761).wrapping_add(seed) % 2000;
            bf16::from_f32((h as f32 / 2000.0 - 0.5) * 0.05)
        })
        .collect()
}

#[test]
fn dspark_graph_matches_eager_bitwise() {
    let _g = gate();
    // CLI-1: the options registry (no env); set before any GPU work.
    gb10_inference::opts::test_set(gb10_inference::opt!("dspark-graph"), Some("1"));
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).unwrap();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let three_d = 3 * cfg.dim;
    let sw = 130usize;

    let top = Dsv4GpuModel::load_trunk_top(&dev, bundle, &cfg).expect("load_trunk_top");
    let embed = top.embed.clone();
    let head = top.head.clone();
    drop(top);
    let mut ds = Dsv4DSpark::load(&dev, bundle, &cfg, 2048, embed, head).expect("Dsv4DSpark::load");

    // Warm the rings (130 positions → t_win saturates at the 128-slot window).
    let warm = dev.htod_sync_copy(&synth_mh(sw, three_d, 7)).expect("htod warm");
    ds.warm(&warm, sw).expect("warm");

    let draft_mh = dev.htod_sync_copy(&synth_mh(1, three_d, 99)).expect("htod draft mh");
    // (start_pos, real_token): capture at 130, replays at growing sp, then short-context
    // replays (t_win < win — the DraftTwin/DraftT/grid patches).
    let mut steps: Vec<(usize, i32)> = (0..8).map(|i| (130 + i * 6, 12345 + i as i32 * 31)).collect();
    steps.push((60, 777));
    steps.push((90, 888));
    steps.extend((0..4).map(|i| (130 + i * 6, 999 + i as i32 * 17)));

    let mut n_logit_cmp = 0usize;
    for (step, (sp, tok)) in steps.iter().copied().enumerate() {
        let (eager_out, eager_logits) = ds.draft_full(&draft_mh, tok, sp).expect("eager draft");
        let (graph_out, graph_logits) = ds.draft_graphed_full(&draft_mh, tok, sp).expect("graphed draft");
        assert_eq!(eager_logits.len(), graph_logits.len(), "logits len");
        let mism = eager_logits
            .iter()
            .zip(graph_logits.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        n_logit_cmp += eager_logits.len();
        eprintln!(
            "[dspark-graph-gate] step {step} sp={sp}: logits bit-mismatches {mism}/{} drafts_e={:?} drafts_g={:?}",
            eager_logits.len(), eager_out.drafts, graph_out.drafts
        );
        assert_eq!(mism, 0, "graphed != eager logits at sp={sp} (step {step})");
        assert_eq!(eager_out.drafts, graph_out.drafts, "draft ids differ at sp={sp}");
    }
    println!("DSPARK-GRAPH-GATE: PASS ({n_logit_cmp} logits compared bitwise over {} steps, eager ≡ graphed)", steps.len());
}
