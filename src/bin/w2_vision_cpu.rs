//! Verify the engine's CPU vision encoder against the HF oracle (merged embeddings rel-L2, CPU).
//! Reads the small test image's reference pixel_values + merger from /tmp dumps (produced by the
//! oracle) and compares `forward_cpu` output. No model run — pure CPU reference validation.

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
        v.push(f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]));
    }
    (v, shape)
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
    // CLI-1: `--data-url <data:...>` (was the W2_DATA_URL env var)
    let argv: Vec<String> = std::env::args().collect();
    // registry options this tool honours through the tower loader: --image-max-edge, --vision-cpu-threads
    gb10_inference::opts::parse_cli(&argv).expect("options");
    // `--data-url <data:...>` or `--data-url-file <FILE>` (a long URL does not fit every command line)
    let data_url = argv.iter().position(|a| a == "--data-url").and_then(|i| argv.get(i + 1).cloned())
        .or_else(|| argv.iter().position(|a| a == "--data-url-file").and_then(|i| argv.get(i + 1))
            .map(|f| std::fs::read_to_string(f).expect("--data-url-file")));
    if let Some(dataurl) = data_url {
        let tower = gb10_inference::vision_tower::VisualTower::load(md).expect("load tower");
        let t = std::time::Instant::now();
        let r = gb10_inference::vision_encoder::process_data_url(&tower, dataurl.trim());
        println!("CPU_TOWER_FORWARD {:.2} s with {} thread(s)", t.elapsed().as_secs_f64(), gb10_inference::vision_encoder::cpu_tower_threads());
        match r {
            Ok(o) => {
                {
                    use sha2::{Digest, Sha256};
                    let mut h = Sha256::new();
                    for v in &o.merged { h.update(v.to_le_bytes()); }
                    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
                    println!("CPU_TOWER_SHA256 {hex}");
                }
                println!("DATA_URL_OK grid=(h={},w={}) num_tokens={} merged[{}] (n*5120={})",
                    o.grid_h, o.grid_w, o.num_tokens, o.merged.len(), o.num_tokens*5120);
                if o.merged.len() == o.num_tokens * 5120 { println!("DATA_URL_SHAPE_OK"); }
            }
            Err(e) => { eprintln!("DATA_URL_ERR: {e}"); println!("DATA_URL_FAIL"); }
        }
        return;
    }
    let tower = gb10_inference::vision_tower::VisualTower::load(md).expect("load tower");
    let (pv, _) = read_f32_raw("/tmp/w2_small_pv.raw"); // [280, 1536]
    let (merger, mshape) = read_f32_raw("/tmp/w2_small_merger.raw"); // [70, 5120]
    let (gh, gw) = (14usize, 20usize);
    let out = tower.forward_cpu(&pv, gh, gw);
    assert_eq!(out.len(), merger.len(), "len rust={} ref={}", out.len(), merger.len());
    let mut maxabs = 0.0f32;
    let mut ss = 0.0f64;
    let mut nn = 0.0f64;
    for i in 0..out.len() {
        let d = (out[i] - merger[i]).abs();
        if d > maxabs { maxabs = d; }
        ss += (out[i] as f64 - merger[i] as f64).powi(2);
        nn += (merger[i] as f64).powi(2);
    }
    let rel_l2 = (ss / (nn + 1e-12)).sqrt();
    println!("VISION_CPU_COMPARE out[{}] ref[{}] maxabs={:.6} rel_L2={:.6}", out.len(), merger.len(), maxabs, rel_l2);
    println!("VISION_CPU shape: rust[{}] ref[{} {:?}]", out.len(), merger.len(), mshape);
    if rel_l2 < 1e-4 {
        println!("VISION_CPU_MATCH");
    } else {
        println!("VISION_CPU_MISMATCH rel_L2={rel_l2:.6}");
    }
}
