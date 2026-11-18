//! P2 oracle gate: the GPU vision tower (`GpuVisualTower`) per-block rel-L2 vs the HF oracle
//! (`tool_probe/w2/refout/tower_states.npz`, the small image grid [1,14,20], 280 patches / 70 tokens).
//!
//! Feeds the SAME HF pixel_values both the CPU-referenced GPU chain and the oracle; the comparison
//! isolates weights+architecture from preproc (PLAN/W2_PREPROC_SPEC.md §10). Passes when every
//! stage (pre_blocks, block_0..26, merger) is inside the family envelope (the CPU chain set
//! rel-L2 2.9e-5; the FP32 GPU chain should be ~3e-5 — the worst block is reported).

fn read_f32_raw(path: &str) -> (Vec<f32>, Vec<usize>) {
    let bytes = std::fs::read(path).expect("read raw");
    let mut shape = Vec::new();
    for off in [0usize, 8] {
        if off + 8 <= bytes.len() {
            shape.push(i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize);
        }
    }
    let data = &bytes[16..];
    let n = data.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(f32::from_le_bytes([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]]));
    }
    (v, shape)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "len mismatch {} vs {}", a.len(), b.len());
    let mut ss = 0.0f64;
    let mut nn = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        ss += d * d;
        nn += (b[i] as f64) * (b[i] as f64);
    }
    (ss / (nn + 1e-12)).sqrt()
}

/// A required `--flag <VALUE>` of this dev tool (no built-in lab path: S-B9-REL-SMALLFIXES item 8).
fn required_flag(name: &str) -> String {
    let argv: Vec<String> = std::env::args().collect();
    argv.iter().position(|a| a == name).and_then(|i| argv.get(i + 1).cloned())
        .unwrap_or_else(|| panic!("usage: pass {name} <DIR>"))
}

fn main() {
    // --model-dir <DIR>: the NVFP4 pack whose vision tower is checked (a 3.8-27b full pack)
    let md_arg = required_flag("--model-dir");
    let md = md_arg.as_str();
    let (pv, _) = read_f32_raw("/tmp/w2_small_pv.raw"); // [280, 1536]
    let (gh, gw) = (14usize, 20usize);
    let n = gh * gw;

    let tower = gb10_inference::vision_tower::VisualTower::load(md).expect("load CPU tower");
    let dev = cudarc::driver::CudaDevice::new(0).expect("device");
    let mut gvt = gb10_inference::vision_gpu::GpuVisualTower::new(dev, &tower).expect("gpu tower");
    let (merged, states) = gvt.forward(&pv, gh, gw, true).expect("gpu forward");

    // states: [pre_blocks, block_0..26]. Reference keys from the oracle npz.
    let mut worst = (0.0f64, String::new());
    let mut all_pass = true;

    let (pre_ref, pre_shape) = read_f32_raw("/tmp/w2_pre_blocks.raw");
    println!("stage pre_blocks : gpu_len={} ref_len={} ({:?})", states[0].len(), pre_ref.len(), pre_shape);
    let r = rel_l2(&states[0], &pre_ref);
    println!("  rel_L2 = {:.3e}", r);
    if states[0].len() != pre_ref.len() { all_pass = false; }
    if r > worst.0 { worst = (r, "pre_blocks".into()); }
    if r > 1e-4 { all_pass = false; }

    for k in 0..27 {
        let (bref, bshape) = read_f32_raw(&format!("/tmp/w2_block_{}.raw", k));
        let label = format!("block_{}", k);
        let r = rel_l2(&states[k + 1], &bref);
        println!("stage {} : gpu_len={} ref_len={} ({:?})  rel_L2 = {:.3e}", label, states[k + 1].len(), bref.len(), bshape, r);
        if states[k + 1].len() != bref.len() { all_pass = false; }
        if r > worst.0 { worst = (r, label); }
        if r > 1e-4 { all_pass = false; }
    }

    let (merg_ref, mshape) = read_f32_raw("/tmp/w2_merger.raw");
    println!("merger      : gpu_len={} ref_len={} ({:?})", merged.len(), merg_ref.len(), mshape);
    let r = rel_l2(&merged, &merg_ref);
    println!("  rel_L2 = {:.3e}", r);
    if merged.len() != merg_ref.len() { all_pass = false; }
    if r > worst.0 { worst = (r, "merger".into()); }
    if r > 1e-4 { all_pass = false; }

    println!("W2_GPU_ORACLE worst={} ({}) n_blocks=27 seq={}", worst.0, worst.1, n);
    if all_pass && worst.0 <= 1e-4 {
        println!("W2_GPU_ORACLE_PASS");
    } else {
        println!("W2_GPU_ORACLE_FAIL worst={:.3e} ({})", worst.0, worst.1);
    }
}
