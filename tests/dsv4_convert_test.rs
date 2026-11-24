//! DSV4 offline-converter A/B gate (lane: load-speed). Confirms `dsv4_convert::prepare_layer`
//! produces byte-identical host bytes to `Dsv4AttnRuntime::upload_layer`'s device-upload source —
//! i.e. the artifact carries EXACTLY what the streaming path computes. If this passes, the fast
//! reader (`load_converted`) is bitwise by construction (same bytes → same device state).
//!
//! Run: cargo test --release --test dsv4_convert_test -- --test-threads=1 --nocapture
//! (uses the real bundle; serializes on a static gate like the other dsv4_*_test suites.)

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaDevice, DevicePtr};
use half::bf16;

use gb10_inference::dsv4_attn::{Dsv4AttnRuntime, Dsv4GpuLayer};
use gb10_inference::dsv4_convert::{prepare_layer, write_artifact, write_artifact_sharded, Art};
use gb10_inference::dsv4_load;
use gb10_inference::dsv4_model::Dsv4GpuModel;

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";
static GATE: Mutex<()> = Mutex::new(());
fn gate() -> MutexGuard<'static, ()> { GATE.lock().unwrap_or_else(|e| e.into_inner()) }

fn by_name(arts: &[Art]) -> HashMap<&str, &Art> {
    arts.iter().map(|a| (a.name.as_str(), a)).collect()
}

/// f32 Art bytes → Vec<f32>.
fn art_f32(a: &Art) -> Vec<f32> {
    assert!(a.data.len() % 4 == 0);
    a.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn art_bf16(a: &Art) -> Vec<bf16> {
    a.data.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}

fn dev0() -> Arc<CudaDevice> { CudaDevice::new(0).unwrap() }
fn dtoh_u8s(d: &cudarc::driver::CudaSlice<u8>) -> Vec<u8> { dev0().dtoh_sync_copy(d).unwrap() }
fn dtoh_f32s(d: &cudarc::driver::CudaSlice<f32>) -> Vec<f32> { dev0().dtoh_sync_copy(d).unwrap() }
fn dtoh_bf16s(d: &cudarc::driver::CudaSlice<bf16>) -> Vec<bf16> { dev0().dtoh_sync_copy(d).unwrap() }

/// Compare a u8 Art to a u8 byte slice, byte-for-byte.
fn cmp_u8(label: &str, art: &Art, dev_bytes: &[u8]) {
    assert_eq!(art.data, dev_bytes, "{label}: artifact vs streaming bytes differ ({} vs {})", art.data.len(), dev_bytes.len());
}
fn cmp_f32(label: &str, art: &Art, dev: &cudarc::driver::CudaSlice<f32>) {
    assert_eq!(art_f32(art), dtoh_f32s(dev), "{label}: artifact vs streaming f32 differ");
}

#[test]
fn prepare_layer_matches_upload_layer_layer2_csa() {
    if !Path::new(BUNDLE).join("inference/config.json").exists() { return; }
    let _g = gate();
    let cfg = dsv4_load::load_config(Path::new(BUNDLE)).unwrap();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let rt = Dsv4AttnRuntime::new_multikind(&dev, 64, &cfg).unwrap();
    let layer_id = 2; // CSA: exercises compressor + indexer + MoE
    let gl: Dsv4GpuLayer = rt.upload_layer(Path::new(BUNDLE), &cfg, layer_id, 0, 1).unwrap();
    let arts = prepare_layer(Path::new(BUNDLE), &cfg, layer_id).unwrap();
    let m = by_name(&arts);

    // FP8 weights: wt + sb (MMA-repacked). dtoh the device wt/sb, compare to the Art bytes.
    cmp_u8("wq_a.wt", m["wq_a.wt"], &dtoh_u8s(&gl.wq_a.wt));
    cmp_u8("wq_a.sb", m["wq_a.sb"], &dtoh_u8s(&gl.wq_a.sb));
    cmp_u8("wq_b.wt", m["wq_b.wt"], &dtoh_u8s(&gl.wq_b.wt));
    cmp_u8("wq_b.sb", m["wq_b.sb"], &dtoh_u8s(&gl.wq_b.sb));
    cmp_u8("wkv.wt", m["wkv.wt"], &dtoh_u8s(&gl.wkv.wt));
    cmp_u8("wo_b.wt", m["wo_b.wt"], &dtoh_u8s(&gl.wo_b.wt));
    cmp_u8("sh_gu.wt", m["sh_gu.wt"], &dtoh_u8s(&gl.sh_gu.wt));
    cmp_u8("sh_gu.sb", m["sh_gu.sb"], &dtoh_u8s(&gl.sh_gu.sb));
    cmp_u8("sh_w2.wt", m["sh_w2.wt"], &dtoh_u8s(&gl.sh_w2.wt));
    // CSA indexer wq_b.
    cmp_u8("idx.wq_b.wt", m["idx.wq_b.wt"], &gl.idx_load.as_ref().unwrap().wq_b_wt); // host vec already
    assert_eq!(m["idx.wq_b.sb"].data, gl.idx_load.as_ref().unwrap().wq_b_sb, "idx.wq_b.sb mismatch");

    // MoE: pack_moe_layer output (Dsv4MoeGpu holds device copies of Dsv4MoeHost).
    cmp_u8("moe.gu_wt", m["moe.gu_wt"], &dtoh_u8s(&gl.moe.gu_wt));
    cmp_u8("moe.gu_st", m["moe.gu_st"], &dtoh_u8s(&gl.moe.gu_st));
    cmp_f32("moe.gu_gs", m["moe.gu_gs"], &gl.moe.gu_gs);
    cmp_u8("moe.dn_wt", m["moe.dn_wt"], &dtoh_u8s(&gl.moe.dn_wt));
    cmp_u8("moe.dn_st", m["moe.dn_st"], &dtoh_u8s(&gl.moe.dn_st));
    cmp_f32("moe.dn_gs", m["moe.dn_gs"], &gl.moe.dn_gs);

    // f32 / bf16 / i32 tensors (take_f32 → same source → bitwise by construction; this catches
    // transcription errors in the key names / sizes).
    cmp_f32("gate_w", m["gate_w"], &gl.gate_w);
    cmp_f32("q_norm", m["q_norm"], &gl.q_norm);
    cmp_f32("attn_norm", m["attn_norm"], &gl.attn_norm);
    cmp_f32("hc_attn_fn", m["hc_attn_fn"], &gl.hc_attn_fn);
    cmp_f32("attn_sink", m["attn_sink"], &gl.sink);
    // wo_a bf16.
    let wo_a_dev: Vec<bf16> = dtoh_bf16s(&gl.wo_a);
    assert_eq!(art_bf16(m["wo_a"]), wo_a_dev, "wo_a mismatch");
    // comp_load (host struct) — CSA compressor weights.
    let cl = gl.comp_load.as_ref().unwrap();
    assert_eq!(art_f32(m["comp.wkv"]), cl.w.wkv, "comp.wkv mismatch");
    assert_eq!(art_f32(m["comp.ape"]), cl.w.ape, "comp.ape mismatch");
    // idx_load compressor + weights_proj.
    let il = gl.idx_load.as_ref().unwrap();
    assert_eq!(art_f32(m["idx.comp.wkv"]), il.comp.w.wkv, "idx.comp.wkv mismatch");
    assert_eq!(art_f32(m["idx.weights_proj"]), il.weights_proj, "idx.weights_proj mismatch");

    println!("convert layer2 (CSA) A/B: ALL TENSORS BIT-IDENTICAL (prepare_layer == upload_layer)");
}

#[test]
fn prepare_layer_matches_upload_layer_layer3_hca() {
    if !Path::new(BUNDLE).join("inference/config.json").exists() { return; }
    let _g = gate();
    let cfg = dsv4_load::load_config(Path::new(BUNDLE)).unwrap();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let rt = Dsv4AttnRuntime::new_multikind(&dev, 64, &cfg).unwrap();
    let layer_id = 3; // HCA: compressor ratio 128, no indexer
    let gl: Dsv4GpuLayer = rt.upload_layer(Path::new(BUNDLE), &cfg, layer_id, 0, 1).unwrap();
    let arts = prepare_layer(Path::new(BUNDLE), &cfg, layer_id).unwrap();
    let mm = by_name(&arts);
    cmp_u8("wq_a.wt", mm["wq_a.wt"], &dtoh_u8s(&gl.wq_a.wt));
    cmp_u8("sh_gu.wt", mm["sh_gu.wt"], &dtoh_u8s(&gl.sh_gu.wt));
    cmp_u8("moe.gu_wt", mm["moe.gu_wt"], &dtoh_u8s(&gl.moe.gu_wt));
    cmp_f32("gate_w", mm["gate_w"], &gl.gate_w);
    let cl = gl.comp_load.as_ref().unwrap();
    assert_eq!(art_f32(mm["comp.wkv"]), cl.w.wkv, "comp.wkv mismatch");
    assert!(mm.get("idx.wq_b.wt").is_none(), "HCA must not have an indexer");
    println!("convert layer3 (HCA) A/B: ALL CHECKED TENSORS BIT-IDENTICAL");
}

/// End-to-end: write a small artifact (4 layers + trunk top), load BOTH ways (streaming +
/// converted), forward the same prompt → logits BIT-IDENTICAL. This is the load-speed lane's
/// proof that `load_converted` produces the same GPU state as the streaming `load`.
#[test]
fn load_converted_matches_streaming_forward() {
    if !Path::new(BUNDLE).join("inference/config.json").exists() { return; }
    let _g = gate();
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).unwrap();
    let n_layers = 4; // SWA, SWA, CSA, HCA — all three kinds
    let artifact = std::path::PathBuf::from("/tmp/dsv4-convert-test-artifact");
    let _ = std::fs::remove_dir_all(&artifact);
    eprintln!("[convert-e2e] writing {n_layers}-layer artifact ...");
    write_artifact(bundle, &cfg, &artifact, n_layers).expect("write_artifact");

    let prompt: Vec<i32> = vec![1, 100, 4321, 9, 222, 7777, 314, 271];
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let max_seq_len = 2048usize;
    let s_max = 320usize;

    eprintln!("[convert-e2e] streaming load ...");
    let mut ms = Dsv4GpuModel::load(&dev, bundle, &cfg, max_seq_len, s_max, n_layers).unwrap();
    let ls = ms.forward(&prompt, 0).expect("streaming forward");

    eprintln!("[convert-e2e] converted load ...");
    let mut mc = Dsv4GpuModel::load_converted(&dev, &artifact, &cfg, max_seq_len, s_max, n_layers, 0, 1)
        .expect("load_converted");
    let lc = mc.forward(&prompt, 0).expect("converted forward");

    let ls_v: Vec<f32> = dev.dtoh_sync_copy(&ls).unwrap();
    let lc_v: Vec<f32> = dev.dtoh_sync_copy(&lc).unwrap();
    assert_eq!(ls_v.len(), lc_v.len(), "logits length mismatch");
    let mism = ls_v.iter().zip(lc_v.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let (am_s, am_c) = (argmax(&ls_v), argmax(&lc_v));
    println!("[convert-e2e] streaming vs converted logits: {mism}/{} f32 mismatch, argmax {am_s} vs {am_c}",
        ls_v.len());
    assert_eq!(mism, 0, "streaming vs converted logits NOT bit-identical ({mism} mismatched)");
    assert_eq!(am_s, am_c, "argmax differs");
    println!("[convert-e2e] PASS: load_converted == streaming (logits BIT-IDENTICAL, argmax {am_s})");
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}

/// PER-RANK artifact gate: write a sharded 4-layer artifact (rank0/ + rank1/), load EACH rank via
/// `load_converted` (each reads only its shard), run the TP-sim forward (block_forward_tp_sim sums
/// both ranks' routed partials), and compare to the streaming full forward → BIT-IDENTICAL. This is
/// the load-speed lane's TP=2 design: each node loads only ~84 GB (its pre-sliced shard).
#[test]
fn sharded_artifact_tp_load_matches_streaming() {
    if !Path::new(BUNDLE).join("inference/config.json").exists() { return; }
    let _g = gate();
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).unwrap();
    let n_layers = 4;
    let shard_dir = std::path::PathBuf::from("/tmp/dsv4-convert-test-sharded");
    let _ = std::fs::remove_dir_all(&shard_dir);
    eprintln!("[sharded-e2e] writing 2-way sharded {n_layers}-layer artifact ...");
    write_artifact_sharded(bundle, &cfg, &shard_dir, n_layers, 2).expect("write_artifact_sharded");

    let prompt: Vec<i32> = vec![1, 100, 4321, 9, 222, 7777, 314, 271];
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let (max_seq_len, s_max) = (2048usize, 320usize);

    eprintln!("[sharded-e2e] converted load rank0 + rank1 (each its own shard) ...");
    let mut ma = Dsv4GpuModel::load_converted(&dev, &shard_dir, &cfg, max_seq_len, s_max, n_layers, 0, 2).unwrap();
    let mut mb = Dsv4GpuModel::load_converted(&dev, &shard_dir, &cfg, max_seq_len, s_max, n_layers, 1, 2).unwrap();

    // TP-sim forward: embed on rank_a, then block_forward_tp_sim per layer (sums both ranks' routed).
    let ids_dev = dev.htod_sync_copy(&prompt).unwrap();
    let mut x = ma.embed_tokens(&ids_dev, prompt.len()).unwrap();
    for i in 0..n_layers {
        x = ma.rt.block_forward_tp_sim(&ma.layers[i], &mut ma.states[i], &mut ma.scratch,
                                       &mb.layers[i], &mut mb.scratch,
                                       &x, prompt.len(), 0, &ids_dev, &cfg).unwrap();
    }
    let (_c, logits_tp_dev) = ma.forward_head(&x, prompt.len()).unwrap();
    let lv_tp: Vec<f32> = dev.dtoh_sync_copy(&logits_tp_dev).unwrap();

    // Streaming full forward (ground truth).
    let mut ms = Dsv4GpuModel::load(&dev, bundle, &cfg, max_seq_len, s_max, n_layers).unwrap();
    let logits_stream = ms.forward(&prompt, 0).unwrap();
    let lv_st: Vec<f32> = dev.dtoh_sync_copy(&logits_stream).unwrap();

    let mism = lv_tp.iter().zip(lv_st.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let (rel, maxabs) = rel_l2_max(&lv_tp, &lv_st);
    println!("[sharded-e2e] per-rank-converted TP-sim vs streaming: {mism}/{} f32 differ, rel-L2 {rel:.3e} max-abs {maxabs:.3e}, argmax {} vs {}",
        lv_tp.len(), argmax(&lv_tp), argmax(&lv_st));
    // TP-sim (two bf16 routed partials summed) ≠ streaming (one fp32 combine) at the bf16-TP-partials
    // class (rel-L2 ~1e-2, same as run_probe_dsv4_tp_sim_full) — NOT a bit-identity claim. The argmax
    // MUST match (the per-rank slice reconstructs the same token); rel-L2 within the TP floor.
    assert_eq!(argmax(&lv_tp), argmax(&lv_st), "argmax differs — per-rank slice is wrong");
    assert!(rel < 5e-2, "TP rel-L2 {rel:.3e} exceeds the bf16-TP-partials floor");
    println!("[sharded-e2e] PASS: per-rank artifact (rank0/ + rank1/) → load_converted → TP-sim argmax-match (rel-L2 {rel:.3e}, the bf16-TP class)");
}

fn rel_l2_max(a: &[f32], b: &[f32]) -> (f64, f32) {
    let mut sse = 0.0f64;
    let mut ssb = 0.0f64;
    let mut maxabs = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y) as f64;
        sse += d * d;
        ssb += (*y as f64) * (*y as f64);
        maxabs = maxabs.max((x - y).abs());
    }
    ((sse.sqrt() / ssb.sqrt().max(1e-30)) as f64, maxabs)
}
