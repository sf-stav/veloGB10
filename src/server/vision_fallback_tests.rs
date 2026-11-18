//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `vision_fallback_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    fn cpu_ok() -> anyhow::Result<crate::vision_encoder::PreparedVision> {
        Ok(crate::vision_encoder::PreparedVision {
            expanded_tokens: vec![1, 2, 3],
            image_embeds: vec![0.25, -0.5, 1.0, 2.0],
            spans: vec![],
        })
    }

    fn gpu_err() -> anyhow::Result<crate::vision_encoder::PreparedVision> {
        Err(anyhow::anyhow!("vision gemm f32: injected cuBLAS failure (--vision-fail-gemm drill)"))
    }

    /// The core PR #16 guarantee: a failing GPU GEMM returns Err (no panic while the mutex is
    /// held — the lock is NOT poisoned) and the request still gets its image via the CPU tower.
    #[test]
    fn gpu_error_falls_back_to_cpu_and_lock_stays_clean() {
        let g = std::sync::Mutex::new(0usize);
        let out = vision_dispatch_with_recovery(&g, |_| gpu_err(), cpu_ok).unwrap();
        assert_eq!(out.image_embeds, vec![0.25, -0.5, 1.0, 2.0]);
        assert!(!g.is_poisoned(), "a returned Err must not poison the tower mutex");
        assert!(g.lock().is_ok(), "the next GPU request must reach the tower again");
    }

    /// A panic in the GPU worker poisons the lock; the NEXT request still runs the GPU path
    /// (into_inner recovery) instead of dying until restart.
    #[test]
    fn poisoned_lock_is_recovered_and_gpu_path_resumes() {
        let g = std::sync::Mutex::new(0usize);
        let gg = &g;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = gg.lock().unwrap();
            panic!("kernel launch boom (drill)");
        }));
        assert!(g.is_poisoned());
        let out = vision_dispatch_with_recovery(&g,
            |t| { *t = 7; cpu_ok() },           // into_inner() gave us the tower back
            || panic!("CPU must not run when the GPU path succeeds")).unwrap();
        assert_eq!(out.image_embeds.len(), 4);
        // S-B9-REL-PRPORT3 (review R5): the successful recovered cycle now clears the poison
        // flag — later requests take the normal lock path (no "poisoned" log every request).
        assert!(!g.is_poisoned(), "a successful recovered GPU cycle must clear the poison flag");
        assert_eq!(*g.lock().unwrap(), 7);
    }

    /// S-B9-REL-PRPORT3 (review R1): the mutex guard must NOT be held across `run_cpu()` — a
    /// slow CPU fallback (53.8 s in the LEG A drill) must not block a concurrent healthy GPU
    /// request: the second request is served by the GPU tower while the first still sits in
    /// its CPU fallback.
    #[test]
    fn cpu_fallback_does_not_block_the_gpu_tower() {
        let g = std::sync::Mutex::new(0usize);
        let (entered, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        let g_ref = &g; // move-closures capture the reference, not the Mutex (Receiver is !Sync)
        std::thread::scope(|s| {
            s.spawn(move || vision_dispatch_with_recovery(g_ref, |_| gpu_err(), move || {
                entered.send(()).unwrap();
                // bound the block so a regression cannot hang the suite forever
                let _ = release_rx.recv_timeout(std::time::Duration::from_secs(10));
                cpu_ok()
            }));
            entered_rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("first request must reach its CPU fallback");
            let (done, done_rx) = std::sync::mpsc::channel();
            let g2 = &g; // the spawn must OWN done (its lifetime is inside the scope body)
            s.spawn(move || {
                let r = vision_dispatch_with_recovery(g2,
                    |t| { *t += 1; cpu_ok() },
                    || panic!("the second request must be served by the GPU tower"));
                let _ = done.send(r.map(|p| p.image_embeds.len()));
            });
            match done_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Ok(n)) => assert_eq!(n, 4, "the GPU path must serve the second request"),
                Ok(Err(e)) => panic!("second request failed: {e:#}"),
                Err(_) => panic!("GPU path blocked behind a concurrent CPU fallback (guard held across run_cpu)"),
            }
            release.send(()).unwrap();
        });
    }

    /// No CPU tower loaded: the GPU error surfaces as a proper Err (HTTP 400), never a
    /// silently image-less 200.
    #[test]
    fn gpu_error_with_no_cpu_tower_is_an_error() {
        let g = std::sync::Mutex::new(0usize);
        let err = vision_dispatch_with_recovery(&g, |_| gpu_err(),
            || Err(anyhow::anyhow!("no vision tower loaded")))
            .err().expect("must be Err");
        assert!(format!("{err:#}").contains("no vision tower loaded"));
        assert!(format!("{err:#}").contains("injected cuBLAS failure"),
                "the GPU cause must stay in the error chain: {err:#}");
    }

    /// The worker-PANIC arm's no-CPU case: the request fails loudly, it is not served
    /// image-less (same message shape the .await unwrap_or_else arm produces).
    /// S-B9-REL-PRPORT3 (reviews R4+R9): the worker-PANIC arm with no CPU tower really errors —
    /// a REAL JoinError from a panicking blocking worker, through the actual fallback helper
    /// (the previous test asserted a string against itself and exercised nothing).
    #[tokio::test]
    async fn panic_fallback_without_cpu_tower_errors_loudly() {
        let panic = tokio::task::spawn_blocking(|| -> usize { panic!("vision worker drill") })
            .await
            .expect_err("a panicking blocking worker must surface as a JoinError");
        // PreparedVision is not Debug: route the Err out by match, not expect_err
        let err = match vision_panic_fallback(None, Vec::new(), Vec::new(), panic).await {
            Ok(_) => panic!("no CPU tower: the request must fail loudly"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("vision worker failed"),
            "the message must name the failing arm: {err:#}"
        );
    }
