//! Vision panic fallback and dispatch-with-recovery for image requests (CPU tower fallback; fails loudly
//! rather than answering image-less). Siblings: `chat.rs` calls these; `mod.rs` has the map.

/// S-B9-REL-PRPORT3 (review R4): the worker-PANIC arm's CPU fallback, on a blocking worker.
/// No CPU tower loaded: the request fails loudly (never a silently image-less 200).
pub(crate) async fn vision_panic_fallback(
    cpu_tower: Option<std::sync::Arc<crate::vision_tower::VisualTower>>,
    urls: Vec<String>,
    toks: Vec<u32>,
    panic: tokio::task::JoinError,
) -> anyhow::Result<crate::vision_encoder::PreparedVision> {
    match cpu_tower {
        Some(tower) => {
            crate::reprintln!("[vision] GPU worker panicked ({panic}), falling back to CPU tower");
            tokio::task::spawn_blocking(move || {
                crate::vision_encoder::prepare_vision_request(&tower, &urls, &toks)
            })
            .await
            .unwrap_or_else(|e2| Err(anyhow::anyhow!(
                "CPU fallback worker failed after the GPU panic ({panic}); {e2}"
            )))
        }
        None => Err(anyhow::anyhow!("vision worker failed: {panic}")),
    }
}

/// PR #16 (ported): GPU-tower dispatch with poisoned-lock recovery and CPU-tower fallback.
/// A cuBLAS/GEMM failure in the GPU tower now returns `Err` (never a panic mid-forward), and
/// the request falls back to the CPU reference tower so the image is never silently dropped
/// (the `len=0` hazard). A lock poisoned by an earlier panic is recovered via `into_inner()`:
/// the CUDA state is still owned by the guard and the next forward re-syncs its stream, so
/// vision keeps working instead of dying until restart (text traffic was never affected).
pub(crate) fn vision_dispatch_with_recovery<T>(
    gpu: &std::sync::Mutex<T>,
    run_gpu: impl FnOnce(&mut T) -> anyhow::Result<crate::vision_encoder::PreparedVision>,
    run_cpu: impl FnOnce() -> anyhow::Result<crate::vision_encoder::PreparedVision>,
) -> anyhow::Result<crate::vision_encoder::PreparedVision> {
    // S-B9-REL-PRPORT3 (review R1): the guard is scoped to `run_gpu` ONLY. The CPU reference
    // tower can take tens of seconds (53.8 s in the LEG A drill); holding the mutex across
    // `run_cpu()` serialized every other image request behind one slow fallback although the
    // GPU tower was free. Drop the guard before the CPU path — it is not under this mutex.
    let gpu_err = {
        let mut recovered = false;
        let mut gvt = match gpu.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                crate::reprintln!("[vision] vision_gpu lock was poisoned, recovering inner tower");
                recovered = true;
                poisoned.into_inner()
            }
        };
        match run_gpu(&mut gvt) {
            Ok(p) => {
                // S-B9-REL-PRPORT3 (review R5): a successful cycle through the recovered tower
                // proves the CUDA state is usable again — clear the poison flag so later requests
                // take the normal lock path instead of logging "poisoned" on every request.
                if recovered {
                    gpu.clear_poison();
                }
                return Ok(p);
            }
            Err(e) => e,
        }
    }; // guard dropped here — run_cpu runs unlocked
    crate::reprintln!("[vision] GPU tower failed ({gpu_err:#}), falling back to CPU tower");
    match run_cpu() {
        Ok(p) => Ok(p),
        // No CPU tower (or it failed too): the request errors out — with the GPU cause
        // in the chain (PR #16: never a silently image-less 200).
        Err(cpu) => Err(cpu.context(format!("GPU tower failed first ({gpu_err:#})"))),
    }
}

