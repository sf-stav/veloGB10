//! Vision cross-check: for ANY model dir with a vision tower, run the CPU reference forward and
//! the GPU tower on the SAME input and compare the merged embeddings (rel-L2). The two paths are
//! independent implementations of the same geometry-driven math; agreement validates the
//! parameterization (dims, rotary, dequant) for towers the V2 oracle never touched.
//! Usage: vision_cross <model-dir> [data-url]  (default: the solid_red fixture re-encoded)
fn main() {
    use base64::Engine;
    use std::io::Write;
    let md = std::env::args().nth(1).expect("usage: vision_cross <model-dir> [data-url]");
    let url = std::env::args().nth(2).unwrap_or_else(|| {
        let png = std::fs::read("/tmp/vision_repro/solid_red.png").expect("fixture png");
        format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png))
    });

    let tower = gb10_inference::vision_tower::VisualTower::load(&md).expect("load tower");
    let pre = {
        // reuse the encoder's preprocess path via a data url (decoded on the host, same input)
        let b64 = url.split("base64,").nth(1).expect("data url");
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim()).expect("b64");
        let img = image::load_from_memory(&bytes).expect("png").to_rgb8();
        let (w, h) = img.dimensions();
        gb10_inference::vision_preproc::preprocess_image(
            h as usize, w as usize, img.as_raw(), &tower.preproc)
    };
    let d = tower.dims;
    eprintln!("dims: hidden={} depth={} inter={} heads={} pos={} patch={} merge={} temporal={} in_ch={} out={}",
        d.hidden, d.depth, d.inter, d.heads, d.num_pos, d.patch, d.temporal, d.merge, d.in_ch, d.out_hidden);

    let cpu = tower.forward_cpu(&pre.pixel_values, pre.grid_h, pre.grid_w);
    eprintln!("cpu  merged[{}] grid=({},{}) tokens={}", cpu.len(), pre.grid_h, pre.grid_w,
        pre.grid_h * pre.grid_w / (d.merge * d.merge));

    let dev = cudarc::driver::CudaDevice::new(0).expect("device");
    let mut gvt = gb10_inference::vision_gpu::GpuVisualTower::new(dev, &tower).expect("gpu tower");
    let (gpu, _states) = gvt.forward(&pre.pixel_values, pre.grid_h, pre.grid_w, false).expect("gpu fwd");
    eprintln!("gpu  merged[{}]", gpu.len());

    assert_eq!(cpu.len(), gpu.len(), "merged len mismatch");
    let mut ss = 0.0f64;
    let mut nn = 0.0f64;
    for i in 0..cpu.len() {
        let dd = cpu[i] as f64 - gpu[i] as f64;
        ss += dd * dd;
        nn += (gpu[i] as f64) * (gpu[i] as f64);
    }
    let rel = (ss / (nn + 1e-12)).sqrt();
    println!("VISION_CROSS rel_l2={:.3e} len={}", rel, cpu.len());
    // Bar: 3e-4 — the fp32 accumulation-order noise class (cuBLAS vs sequential CPU; measured
    // 1.6e-5..1.2e-4 across tower geometries; hd=64 towers sit at ~1e-4). This is a CONSISTENCY
    // check between the two independent implementations; the semantic gate is the end-to-end
    // image/color check (the width/rotary/dequant parameterization would blow this up by orders
    // of magnitude if wrong).
    if rel <= 3e-4 {
        println!("VISION_CROSS_PASS");
    } else {
        println!("VISION_CROSS_FAIL");
        std::process::exit(1);
    }
    let _ = std::io::stdout().flush();
}
