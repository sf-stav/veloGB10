//! `probe_bandwidth`: STREAM-style pure-read bandwidth probe (the roofline every other number is measured against).
//! Needs no model: it loads only `bw_read_b` on a bare device. Map: `mod.rs`.

use super::*;

/// STREAM-style pure-read bandwidth. This is the roofline every other number is measured against,
/// and we had been quoting two different values for it (248 GB/s vs 216 GB/s observed) — a 15%
/// spread that decides whether the GEMMs have 10% left in them or 25%, and whether a competitor's
/// claimed tok/s is even physically possible on this part. So measure it, don't quote it.
///
/// Needs no model: it loads only `bw_read_b` (build-ID checked) on a bare device. It used to load a
/// whole NVFP4 GpuModel first, which panicked on EXL3 packs (qwen4_exp) and then hung holding the GPU.
///
/// `sustained = Some(seconds)` (§2.3 audit): hammer memory bandwidth CONTINUOUSLY for `seconds`,
/// reporting achieved GB/s per ~2s window. If sustained bandwidth sags vs the cold peak, LPDDR5x is
/// thermally derating under load — which would cap every roofline number (decode, TP2, Hy3).
pub fn probe_bandwidth(sustained: Option<u64>) -> anyhow::Result<()> {
    let dev = CudaDevice::new(0)?;
    let bptx = Ptx::from_src(crate::ptx_embed::GPU_BATCH);
    dev.load_ptx(bptx, "gpu_batch", &["bw_read_b", "kernel_build_id"])?;
    GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
    let f = dev.get_func("gpu_batch", "bw_read_b").ok_or_else(|| anyhow::anyhow!("bw_read_b missing"))?;

    let gib: usize = 8;   // large enough that no cache can hold it
    let n4 = gib * 1024 * 1024 * 1024 / 16;            // uint4 elements
    let buf = dev.alloc_zeros::<u8>(n4 * 16)?;
    let sink = dev.alloc_zeros::<f32>(1)?;
    dev.synchronize()?;
    let bytes = (n4 * 16) as f64;
    let run = |blocks: u32| -> anyhow::Result<()> {
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { f.clone().launch(cfg, (&sink, *buf.device_ptr() as u64, n4 as i64))? };
        Ok(())
    };

    if let Some(seconds) = sustained {
        run(8192)?; dev.synchronize()?; // warm
        println!("=== SUSTAINED bandwidth ({} s, 8 GiB reads back-to-back) — watch for thermal sag ===", seconds);
        println!("  {:>6}  {:>8}  {:>8}", "t(s)", "GB/s", "min-so-far");
        let start = std::time::Instant::now();
        let (mut peak, mut trough) = (0f64, f64::INFINITY);
        while start.elapsed().as_secs() < seconds {
            let w0 = std::time::Instant::now();
            let mut iters = 0u32;
            // run ~2s worth of back-to-back reads
            while w0.elapsed().as_secs_f64() < 2.0 { run(8192)?; iters += 1; }
            dev.synchronize()?;
            let s = w0.elapsed().as_secs_f64();
            let gbs = bytes * iters as f64 / s / 1e9;
            peak = peak.max(gbs); trough = trough.min(gbs);
            println!("  {:>6.0}  {:>8.1}  {:>8.1}", start.elapsed().as_secs_f64(), gbs, trough);
        }
        println!("  PEAK {:.1} GB/s  TROUGH {:.1} GB/s  SAG {:.1}%  ({})",
                 peak, trough, 100.0*(peak-trough)/peak,
                 if (peak-trough)/peak < 0.03 { "STABLE — no thermal derating" } else { "SAG — sustained < cold" });
        return Ok(());
    }

    println!("=== pure-read bandwidth (STREAM-style, {} GiB, 16-byte vectorized loads) ===", gib);
    let mut best = 0.0f64;
    for blocks in [1024u32, 2048, 4096, 8192] {
        run(blocks)?; dev.synchronize()?;                  // warm
        let t0 = std::time::Instant::now();
        for _ in 0..5 { run(blocks)?; dev.synchronize()?; }
        let s = t0.elapsed().as_secs_f64() / 5.0;
        let gbs = bytes / s / 1e9;
        if gbs > best { best = gbs; }
        println!("  {:5} blocks x 256 thr   {:7.2} ms   {:6.1} GB/s", blocks, s * 1000.0, gbs);
    }
    println!("  PEAK OBSERVED: {:.0} GB/s   (GB10 LPDDR5x theoretical ~273 GB/s => {:.0}% of peak)",
             best, 100.0 * best / 273.0);
    if best < 230.0 {
        println!("  NOTE: an IDLE GB10 reads at ~238 GB/s. A lower figure means something else was");
        println!("        using memory bandwidth (a leftover server? nsys? a build?). Unified LPDDR5x");
        println!("        is shared with the CPU. Re-run on an idle machine before trusting it --");
        println!("        a contended roofline once read 234 and skewed every efficiency number.");
    }
    Ok(())
}
