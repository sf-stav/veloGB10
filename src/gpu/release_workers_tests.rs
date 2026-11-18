//! Unit tests for the `ReleaseWorkers` guard (former inline test module `release_workers_tests` of `gpu.rs`; file map in `mod.rs`).

    use super::ReleaseWorkers;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex};

    /// The loader's worker shape: pop a job or wait on the condvar until `done`.
    fn worker(jq: &(Mutex<VecDeque<u32>>, Condvar), done: &AtomicBool) {
        loop {
            let job = {
                let mut g = jq.0.lock().unwrap();
                loop {
                    if let Some(j) = g.pop_front() { break Some(j); }
                    if done.load(Ordering::Relaxed) { break None; }
                    g = jq.1.wait(g).unwrap();
                }
            };
            if job.is_none() { break; }
        }
    }

    type Jq = Arc<(Mutex<VecDeque<u32>>, Condvar)>;

    /// Spawn a thread that runs `scope { 4 workers; body panics }`, with or without the guard.
    /// Returns the result channel and the handles needed to release the workers by hand.
    fn run_scope(with_guard: bool) -> (std::sync::mpsc::Receiver<bool>, Jq, Arc<AtomicBool>) {
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        let jq: Jq = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let done = Arc::new(AtomicBool::new(false));
        let (jq_t, done_t) = (jq.clone(), done.clone());
        std::thread::spawn(move || {
            let (jq, done) = (jq_t, done_t);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                std::thread::scope(|s| {
                    for _ in 0..4 {
                        let (jq, done) = (jq.clone(), done.clone());
                        s.spawn(move || worker(&jq, &done));
                    }
                    // let the workers park on the condvar before the body panics
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let _release = if with_guard {
                        Some(ReleaseWorkers(|| {
                            let _held = jq.0.lock().unwrap_or_else(|e| e.into_inner());
                            done.store(true, Ordering::Relaxed);
                            jq.1.notify_all();
                        }))
                    } else { None };
                    panic!("simulated shard-loop panic (bytemuck alignment)");
                })
            }));
            let _ = tx.send(r.is_err());
        });
        (rx, jq, done)
    }

    /// A panic in the scope body returns promptly WITH the guard. Runs on a watchdog thread so a
    /// regression fails the test instead of hanging the suite.
    #[test]
    fn a_panic_in_the_scope_body_releases_the_workers() {
        let (rx, _jq, _done) = run_scope(true);
        match rx.recv_timeout(std::time::Duration::from_secs(20)) {
            Ok(panicked) => assert!(panicked, "the body's panic must still propagate (not be swallowed)"),
            Err(_) => panic!("scope did not return: the workers were never released (the hang)"),
        }
    }

    /// NEGATIVE CONTROL: the pre-fix shape (no guard) really does hang — so the test above can
    /// tell the bug from the fix. The parked workers are released by hand afterwards.
    #[test]
    fn without_the_guard_the_scope_hangs_the_original_bug() {
        let (rx, jq, done) = run_scope(false);
        assert!(rx.recv_timeout(std::time::Duration::from_secs(3)).is_err(),
                "control failed: the unguarded scope returned, so this harness cannot see the hang");
        {
            let _held = jq.0.lock().unwrap();
            done.store(true, Ordering::Relaxed);
            jq.1.notify_all();
        }
        let panicked = rx.recv_timeout(std::time::Duration::from_secs(20)).expect("released scope returns");
        assert!(panicked);
    }

    /// The normal path is unchanged: the guard fires once at the end and is harmless.
    #[test]
    fn the_normal_path_still_completes() {
        let jq = Arc::new((Mutex::new(VecDeque::<u32>::new()), Condvar::new()));
        let done = Arc::new(AtomicBool::new(false));
        std::thread::scope(|s| {
            for _ in 0..3 {
                let (jq, done) = (jq.clone(), done.clone());
                s.spawn(move || worker(&jq, &done));
            }
            let _release = ReleaseWorkers(|| {
                let _held = jq.0.lock().unwrap_or_else(|e| e.into_inner());
                done.store(true, Ordering::Relaxed);
                jq.1.notify_all();
            });
            for j in 0..10 { jq.0.lock().unwrap().push_back(j); jq.1.notify_all(); }
        });
        // reaching this line IS the claim: the scope joined (no hang) after the guard released the workers
        assert!(done.load(Ordering::Relaxed));
    }
