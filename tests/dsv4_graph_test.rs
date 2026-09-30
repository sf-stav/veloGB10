//! CUDA-graph decode gate (R3A.1/E2): the graphed decode step must reproduce the EAGER
//! decode step BITWISE — same kernels, same args, same state machine; only the launch
//! vehicle changes. Drives both paths component-level (--graph set for BOTH so the
//! SEQ-arm capacity allocs are identical — the capacity arm is size-only, value-neutral):
//!   eager   = forward_streams + forward_head per token (the raw eager path);
//!   graphed = Dsv4GpuModel::forward_decode_graphed per token (lazy V0/V4/V128 captures).
//! 64 tokens from a 192-token prompt ⇒ covers all three fire variants (V128 at sp=255)
//! and all three layer kinds (n_layers=4 = [SWA,SWA,CSA,HCA]). Compares per-token logits
//! f32-BITWISE and the greedy id chain. State snapshot/restore between the two runs uses
//! the DSpark verify-state machinery (kv_cache + compressor + indexer).

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::CudaDevice;

use gb10_inference::{dsv4_load, dsv4_model::Dsv4GpuModel};

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";

fn gate() -> MutexGuard<'static, ()> {
    static G: Mutex<()> = Mutex::new(());
    G.lock().unwrap_or_else(|e| e.into_inner())
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}

#[test]
fn graph_decode_matches_eager_bitwise() {
    let _g = gate();
    // BOTH arms see the flag: the SEQ-arm idxs capacity allocs must match (size-only).
    // CLI-1: the options registry (no env); set before any GPU work.
    gb10_inference::opts::test_set(gb10_inference::opt!("graph"), Some("1"));
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).unwrap();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let (max_seq_len, s_max, n_layers) = (2048usize, 320usize, 4usize);
    let prompt_len = 192usize;
    let n_tok = 64usize;
    let prompt: Vec<i32> = (0..prompt_len).map(|i| ((7 + i as i64 * 9973) % cfg.vocab_size as i64) as i32).collect();

    let mut m = Dsv4GpuModel::load(&dev, bundle, &cfg, max_seq_len, s_max, n_layers).unwrap();

    // ---- prefill (eager) + snapshot ----
    let mut logits = m.forward(&prompt, 0).expect("prefill");
    let snaps = m.snapshot_verify_state().expect("snapshot");

    // ---- eager reference: raw forward_streams + forward_head per token ----
    let mut ids_eager = Vec::with_capacity(n_tok);
    let mut logits_eager: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    let mut pos = prompt_len;
    for _ in 0..n_tok {
        let tok = argmax(&dev.dtoh_sync_copy(&logits).unwrap()) as i32;
        ids_eager.push(tok);
        let x = m.forward_streams(&[tok], pos).expect("eager streams");
        let (_c, l) = m.forward_head(&x, 1).expect("eager head");
        logits_eager.push(dev.dtoh_sync_copy(&l).unwrap());
        logits = l;
        pos += 1;
    }

    // ---- restore + graphed ----
    m.restore_verify_state(&snaps).expect("restore");
    let mut ids_graph = Vec::with_capacity(n_tok);
    let mut pos = prompt_len;
    // `logits` after restore is the last EAGER step's stale buffer; the greedy chain
    // restarts from the snapshot's post-prefill state, whose argmax is ids_eager[0].
    let mut tok = ids_eager[0];
    let mut logits_graph_prev: Vec<f32> = Vec::new();
    for i in 0..n_tok {
        if i > 0 {
            tok = argmax(&dev.dtoh_sync_copy(&logits).unwrap()) as i32;
        }
        ids_graph.push(tok);
        logits = m.forward_decode_graphed(tok, pos).expect("graphed forward");
        let lg: Vec<f32> = dev.dtoh_sync_copy(&logits).unwrap();
        let mism = lg.iter().zip(logits_eager[i].iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("[graph-gate] tok {i} (sp={pos}): logits bit-mismatches {mism}/{}", lg.len());
        if mism > 0 && i > 0 {
            let prev_same = lg.iter().zip(logits_graph_prev.iter()).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            eprintln!("[graph-gate] DIAG: graphed[{i}] == graphed[{}] on {prev_same}/{} values; g1[:4]={:?} e1[:4]={:?}",
                i - 1, lg.len(),
                &lg[..4], &logits_eager[i][..4]);
        }
        logits_graph_prev = lg.clone();
        assert_eq!(mism, 0, "graphed != eager at token {i} (sp={pos}): {mism} f32 mismatches");
        pos += 1;
    }
    assert_eq!(ids_graph, ids_eager, "greedy id chains diverge");
    eprintln!("[graph-gate] PASS: graphed ≡ eager bitwise over {n_tok} tokens (V0+V4+V128)");
}
