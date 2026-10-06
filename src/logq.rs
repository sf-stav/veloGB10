//! LR-1 (REL_LONGRUN audit, v0.7.3 hardening): a bounded, NON-BLOCKING queue for per-request
//! log lines that run on the scheduler thread (and on single-threaded tokio handler runtimes).
//!
//! Problem being fixed: ~10-20 unconditional `println!/eprintln!` per request execute on the
//! scheduler thread. When stdout/stderr is a pipe (journald, docker, `| tee`) whose reader stops,
//! the kernel pipe buffer (64 KiB) fills after ~32 requests and the NEXT write BLOCKS the
//! scheduler thread mid-round: the engine freezes with no error line — to a user it looks exactly
//! like "inference stopped". See PLAN/REL_LONGRUN_REPORT.md LR-1.
//!
//! Fix: the printing thread only does a `try_send` into a bounded queue (4096 lines); ONE writer
//! thread does the blocking writes. When the queue is full the line is DROPPED and counted (the
//! queue drains in milliseconds under any healthy reader, so drops only happen when the reader
//! really is stalled — at which point new log lines are worthless anyway); when the writer
//! recovers it prints one "N log lines dropped (reader stalled)" summary line.
//!
//! What MUST stay direct (NOT routed through here): boot banners, FATAL/panic/abort lines and
//! anything on a process-exit path — those must appear even if this module is wedged, and
//! process::exit() does not wait for the writer thread. `flush()` exists for exit paths that can
//! afford a bounded wait (the exit(70) sites already sleep 300 ms to flush error events).
//!
//! Ordering caveat (accepted, documented): lines routed through the queue can interleave
//! differently with DIRECT prints from other threads. Lines among themselves keep FIFO order.

use std::sync::atomic::{AtomicIsize, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender, TrySendError, Receiver};
use std::sync::OnceLock;
use std::time::Duration;

/// Queue capacity: lines. ~100 KiB worst case; a healthy reader drains this in milliseconds.
pub const CAP: usize = 4096;

static DROPPED: AtomicU64 = AtomicU64::new(0);
static QUEUED: AtomicIsize = AtomicIsize::new(0);
static TX: OnceLock<SyncSender<Line>> = OnceLock::new();

/// One routed log line. `err` = it was an `eprintln!` (stderr) rather than `println!` (stdout),
/// so the writer emits to the same stream the original line went to (text unchanged).
pub struct Line {
    pub err: bool,
    pub text: String,
}

/// Total lines dropped because the queue was full (the reader was stalled). Exposed as the
/// `velogb10_log_lines_dropped_total` gauge.
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Core enqueue, shared by the global sender and the unit tests so the tested path IS the
/// production path. NEVER blocks: on a full queue the line is dropped and counted.
pub fn try_line(tx: &SyncSender<Line>, dropped: &AtomicU64, err: bool, text: String) {
    match tx.try_send(Line { err, text }) {
        Ok(()) => {
            QUEUED.fetch_add(1, Ordering::Relaxed);
        }
        Err(TrySendError::Full(_)) => {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
        // Writer thread is gone (it only dies at process exit): fall back to a direct write so
        // the line is still not LOST — the stall risk returns only in that impossible window.
        Err(TrySendError::Disconnected(l)) => {
            if l.err { eprintln!("{}", l.text); } else { println!("{}", l.text); }
        }
    }
}

/// H10 (REL_V0_7_3): the writer's emit. A panicking `println!` on a broken/stalled pipe would
/// KILL the writer thread (QUEUED stuck > 0, every later flush waits its full timeout, and the
/// try_line fallback takes the blocking path on every line). Write with IGNORED errors
/// instead: the pipe may recover, and if it never does, the enqueue-side drop counter is the
/// record. Same output bytes as println!/eprintln! (stdout is line-buffered; stderr raw).
pub fn emit_line(l: &Line) {
    use std::io::Write;
    if l.err {
        let mut h = std::io::stderr().lock();
        let _ = h.write_all(l.text.as_bytes()).and_then(|_| h.write_all(b"\n"));
    } else {
        let mut h = std::io::stdout().lock();
        let _ = h.write_all(l.text.as_bytes()).and_then(|_| h.write_all(b"\n"));
    }
    QUEUED.fetch_sub(1, Ordering::Relaxed);
}

fn sender() -> &'static SyncSender<Line> {
    TX.get_or_init(|| {
        let (tx, rx) = sync_channel::<Line>(CAP);
        std::thread::Builder::new()
            .name("logq-writer".into())
            .spawn(move || {
                writer_loop(
                    &rx,
                    &DROPPED,
                    &emit_line,
                    &|n: u64| eprintln!(
                        "[logq] recovered: {n} log lines were dropped while the log reader was stalled (velogb10_log_lines_dropped_total)"
                    ),
                )
            })
            .expect("logq writer thread");
        tx
    })
}

/// The writer thread body: blocking writes happen HERE, never on the sender's thread. Every
/// drain/idle boundary re-checks the drop counter so the recovery report lands promptly.
pub fn writer_loop(
    rx: &Receiver<Line>,
    dropped: &AtomicU64,
    emit: &dyn Fn(&Line),
    report: &dyn Fn(u64),
) {
    let mut reported = 0u64;
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(l) => {
                emit(&l);
                report_drops(dropped, &mut reported, report);
            }
            Err(RecvTimeoutError::Timeout) => report_drops(dropped, &mut reported, report),
            Err(RecvTimeoutError::Disconnected) => {
                report_drops(dropped, &mut reported, report);
                break;
            }
        }
    }
}

fn report_drops(dropped: &AtomicU64, reported: &mut u64, report: &dyn Fn(u64)) {
    let d = dropped.load(Ordering::Relaxed);
    if d > *reported {
        report(d - *reported);
        *reported = d;
    }
}

/// Route one line through the global queue (what the `rprintln!`/`reprintln!` macros call).
pub fn send(err: bool, text: String) {
    try_line(sender(), &DROPPED, err, text);
}

/// Bounded wait until every queued line has been WRITTEN (not merely dequeued): for exit paths
/// that can afford it, so the last log lines are not lost to `process::exit`.
pub fn flush(timeout: Duration) {
    let t0 = std::time::Instant::now();
    while QUEUED.load(Ordering::Relaxed) > 0 && t0.elapsed() < timeout {
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// H7 (REL_V0_7_3 review): serving-path exits lose the queued log lines — a bare
/// `process::exit` kills the writer thread mid-pipe. Flush with a bounded wait, THEN exit.
/// FATAL messages stay DIRECT (they are printed before this is called); this only protects
/// the queued per-request lines.
pub fn flush_and_exit(timeout: Duration, code: i32) -> ! {
    flush(timeout);
    std::process::exit(code)
}

/// Per-request / per-round log line -> bounded queue, stdout variant. Text is IDENTICAL to the
/// `println!` it replaces; only the delivery is non-blocking.
#[macro_export]
macro_rules! rprintln {
    ($($arg:tt)*) => { $crate::logq::send(false, format!($($arg)*)) };
}

/// Per-request / per-round log line -> bounded queue, stderr variant.
#[macro_export]
macro_rules! reprintln {
    ($($arg:tt)*) => { $crate::logq::send(true, format!($($arg)*)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Barrier, Mutex};
    use std::time::Instant;

    /// LR-1 proof: a writer BLOCKED forever must not block the sender; overflow drops+counts.
    /// Fills a tiny queue whose consumer is parked, asserts every enqueue returns immediately,
    /// the drop counter rises by exactly the overflow, and the writer is still blocked.
    #[test]
    fn blocked_writer_never_blocks_sender_and_counts_drops() {
        let (tx, rx) = sync_channel::<Line>(2);
        let dropped = AtomicU64::new(0);
        // Deterministic ordering: the consumer takes EXACTLY one line, tells us, then parks.
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let parked = Arc::new(Barrier::new(2));
        let parked2 = Arc::clone(&parked);
        let release = Arc::new(Barrier::new(2));
        let release2 = Arc::clone(&release);
        let consumer = std::thread::spawn(move || {
            let first = rx.recv_timeout(Duration::from_secs(5)).expect("first line");
            assert_eq!(first.text, "line 0", "FIFO: the first line goes first");
            go_tx.send(()).expect("signal");
            parked2.wait(); // "reader stalled": parked while the sender keeps trying
            release2.wait();
        });
        try_line(&tx, &dropped, false, "line 0".to_string());
        go_rx.recv_timeout(Duration::from_secs(5)).expect("consumer holds line 0");
        // Queue is empty (cap 2): two lines land, three drop — and none of it may block.
        let t0 = Instant::now();
        for i in 1..6 {
            try_line(&tx, &dropped, false, format!("line {i}"));
        }
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "try_line blocked for {elapsed:?}; LR-1 exists so that it never does"
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 3, "cap 2 holds 2, drops 3");
        assert!(!consumer.is_finished(), "the stalled reader must still be parked, not finished");
        parked.wait();
        assert_eq!(dropped.load(Ordering::Relaxed), 3, "no further drops while parked");
        release.wait();
        consumer.join().unwrap();
    }

    /// LR-1 proof #2: the production `writer_loop` drains the queued capacity in FIFO order
    /// and reports the dropped count exactly once, then exits cleanly on disconnect.
    #[test]
    fn writer_loop_drains_in_order_and_reports_drops_once() {
        let (tx, rx) = sync_channel::<Line>(4);
        let dropped = AtomicU64::new(0);
        for i in 0..7 {
            try_line(&tx, &dropped, i % 2 == 0, format!("l{i}"));
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 3, "cap 4 holds 4, drops 3");
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let reports: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let (s2, r2) = (Arc::clone(&seen), Arc::clone(&reports));
        let writer = std::thread::spawn(move || {
            let emit = |l: &Line| s2.lock().unwrap().push(l.text.clone());
            let report = |n: u64| r2.lock().unwrap().push(n);
            writer_loop(&rx, &dropped, &emit, &report);
        });
        drop(tx); // disconnect => drain, report, exit
        writer.join().unwrap();
        let got = seen.lock().unwrap().clone();
        assert_eq!(got, vec!["l0", "l1", "l2", "l3"], "FIFO order preserved for queued lines");
        assert_eq!(*reports.lock().unwrap(), vec![3u64], "one summary report of the exact count");
    }
}

#[cfg(test)]
/// H5 (REL_V0_7_3 review + the GATES1 finding): Rust's println!/eprintln! take the
/// process-wide stdout/stderr lock; while the logq writer thread is blocked in write(2) on a
/// stalled pipe it HOLDS that lock, and EVERY remaining direct print from ANY thread (the
/// scheduler, the HTTP runtime, TP threads) blocks too. This test pins the per-file count of
/// direct prints outside #[cfg(test)] modules against a committed baseline: a NEW direct
/// print anywhere fails here until its author either converts it to rprintln!/reprintln! or
/// CONSCIOUSLY updates this table (and says why in the commit).
///
/// Baseline provenance: re-baselined 2026-11-06 by S-B9-REL-FIXC with the DIRECT-ONLY
/// counting (println!/eprintln! minus rprintln!/reprintln!), on master 32fb338 AFTER both
/// sweeps (FIXA serving-surface + FIXB engine-side) — the FIXB-sweep-pending marker is gone
/// and these are the real post-sweep direct counts. What may legitimately stay direct:
/// boot/one-time banners, probe/bench/diagnostic-only entry points, FATAL/exit-path lines
/// (logq's own writer and fallback).
mod h5_direct_print_baseline_tests {
    /// Counting rule (S-B9-REL-FIXC): strip every #[cfg(test)] mod (brace-matched), then
    /// count DIRECT prints only — println! and eprintln! count one each; the logq macros
    /// rprintln!/reprintln! (crate::-qualified or not) count ZERO. The old body counted the
    /// raw "println!(" substring, which the macros also contain, so converting a steady-state
    /// println! to rprintln! left the count UNCHANGED and a reverted site went undetected.
    fn count_direct_prints(src: &str) -> u32 {
        let s = src.as_bytes();
        let mut kept = String::with_capacity(src.len());
        let mut i = 0usize;
        while i < s.len() {
            let rest = &src[i..];
            let Some(at) = rest.find("#[cfg(test)]") else { kept.push_str(rest); break };
            let after_attr = &rest[at + 12..];
            let Some(brace) = after_attr.find('{') else { kept.push_str(rest); break };
            // FIXC: doc comments may sit between #[cfg(test)] and the mod item (this very
            // module does that). Decide on the LAST non-empty line before the brace: it must
            // be `mod NAME` (optionally `pub`/`pub(crate)`-prefixed); anything else (fn, use,
            // a plain expression) means the attribute is not a test-module opener.
            let between = after_attr[..brace].trim();
            let last = between.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
            let mut t = last.trim().strip_prefix("pub").unwrap_or(last.trim()).trim();
            if let Some(r) = t.strip_prefix("(crate)") { t = r.trim(); }
            if !t.starts_with("mod ") { kept.push_str(&rest[..at + 12]); i += at + 12; continue; }
            // brace-match the module body (skipping string literals)
            let open = i + at + 12 + brace;
            let mut depth = 0usize;
            let mut j = open;
            let mut in_str = false;
            let mut esc = false;
            let b = src.as_bytes();
            while j < b.len() {
                let c = b[j] as char;
                if esc { esc = false; }
                else if c == '\\' { esc = true; }
                else if c == '"' { in_str = !in_str; }
                else if !in_str {
                    if c == '{' { depth += 1; }
                    else if c == '}' {
                        depth -= 1;
                        if depth == 0 { break; }
                    }
                }
                j += 1;
            }
            kept.push_str(&src[i..i + at]);
            i = j + 1;
        }
        // FIXC: subtract the logq macro occurrences ("reprintln!(" does NOT contain
        // "rprintln!(", so the two subtractions are independent; each macro occurrence
        // contributes exactly one "println!(" match, so this cannot underflow — saturating
        // anyway). Conservative TEXTUAL count by design: a print call mentioned inside an
        // ordinary string literal or doc line still counts (over-counting is safe,
        // under-counting is not).
        let total = kept.matches("println!(").count() as u32;
        let via_macros = kept.matches("rprintln!(").count() as u32
            + kept.matches("reprintln!(").count() as u32;
        total.saturating_sub(via_macros)
    }

    /// FIXC: the counter counts DIRECT prints only — the logq macros count zero.
    #[test]
    fn counter_counts_direct_prints_only() {
        assert_eq!(count_direct_prints(r#"fn a() { println!("a"); }"#), 1);
        assert_eq!(count_direct_prints(r#"fn a() { eprintln!("a"); }"#), 1);
        assert_eq!(count_direct_prints(r#"fn a() { rprintln!("a"); }"#), 0);
        assert_eq!(count_direct_prints(r#"fn a() { reprintln!("a"); }"#), 0);
        assert_eq!(count_direct_prints(r#"fn a() { crate::rprintln!("a"); }"#), 0);
        assert_eq!(count_direct_prints(r#"fn a() { crate::reprintln!("a"); }"#), 0);
    }

    /// FIXC: mixed sources, cfg(test) stripping, and the mutation the old counter missed —
    /// an rprintln! site REVERTED to println! must move the count by one.
    #[test]
    fn counter_handles_mixed_sources_and_test_modules() {
        let mixed = r#"
            fn a() { println!("one"); }
            fn b() { crate::reprintln!("two"); }
            fn c() { eprintln!("three"); }
            fn d() { crate::rprintln!("four"); }
            #[cfg(test)]
            mod tests {
                #[test]
                fn t() { println!("not counted"); }
            }
            fn e() { println!("five"); }
        "#;
        assert_eq!(count_direct_prints(mixed), 3,
            "the two direct prints + eprintln count; the logq macros and the cfg(test) mod count zero");
        // the GATES1 class: a periodic line on the scheduler thread, converted, then reverted
        assert_eq!(count_direct_prints(r#"fn s() { rprintln!("[spec-pass]"); }"#), 0);
        assert_eq!(count_direct_prints(r#"fn s() { println!("[spec-pass]"); }"#), 1,
            "reverting one macro site must move the count by one");
    }

    /// FIXC: documented limitation — the count is conservatively textual.
    #[test]
    fn counter_is_conservatively_textual() {
        // a print call MENTIONED inside an ordinary string literal counts (safe direction)
        assert_eq!(count_direct_prints(r#"let s = "call println!(x) here";"#), 1);
        assert_eq!(count_direct_prints("/// see println!( above"), 1);
        // macro mentions in literals/docs subtract like real macro calls: still zero-ish,
        // never negative
        assert_eq!(count_direct_prints("/// rprintln!( docs"), 0);
    }

    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { walk(&p, out); }
                else if p.extension().is_some_and(|x| x == "rs") { out.push(p.display().to_string()); }
            }
        }
    }

    /// Baseline snapshot 2026-11-06 (rel/fixc): DIRECT-ONLY counts, post both sweeps.
    #[test]
    fn direct_print_counts_match_the_committed_baseline() {
        assert!(std::path::Path::new("src").is_dir(),
            "run from the repo root (the vision strict-load test has the same convention)");
        let baseline: &[(&str, u32)] = &[
            ("src/batch.rs", 35),
            ("src/bin/dsv4_attn_replay.rs", 9),
            ("src/bin/dsv4_convert.rs", 3),
            ("src/bin/dsv4_replay.rs", 10),
            ("src/bin/dsv4_stage_debug.rs", 2),
            ("src/bin/vision_cross.rs", 6),
            ("src/bin/w2_preproc_golden.rs", 8),
            ("src/bin/w2_vision_cpu.rs", 8),
            ("src/bin/w2_vision_gpu.rs", 8),
            ("src/bin/w2_vision_load.rs", 6),
            ("src/cluster.rs", 56),
            ("src/dflash.rs", 2),
            ("src/dflash2/round.rs", 7),
            ("src/dflash2/stepdump.rs", 4),
            ("src/dspark/round.rs", 4),
            ("src/dsv4_attn.rs", 1),
            ("src/dsv4_convert.rs", 9),
            ("src/dsv4_dspark.rs", 7),
            ("src/dsv4_graph.rs", 5),
            ("src/dsv4_model.rs", 13),
            ("src/engine.rs", 1),
            ("src/exl3.rs", 17),
            ("src/exl3_autotune.rs", 15),
            ("src/exl3_bench.rs", 123),
            ("src/exl3_forward.rs", 317), // rel/cutb: 318 -> 317 (EXL3_GRAPH packed-missing marker -> rprintln!)
            ("src/exl3_forward/dense.rs", 12),
            ("src/exl3_forward/dhead.rs", 11),
            ("src/exl3_forward/w4s.rs", 8),
            ("src/exl3_forward/wp16.rs", 12),
            ("src/exl3_forward/xtp.rs", 74),
            ("src/exl3_kvnll.rs", 9),
            ("src/exl3_serve.rs", 85),
            ("src/exl3_tune.rs", 7),
            ("src/exl3_wp27.rs", 8),
            ("src/gptq.rs", 42),
            ("src/gpu.rs", 323),
            ("src/json_schema.rs", 1),
            ("src/kernels.rs", 3),
            ("src/logq.rs", 3),
            ("src/main.rs", 1552),
            ("src/memwatch.rs", 3),
            ("src/model.rs", 3),
            ("src/net.rs", 13),
            ("src/opts.rs", 1),
            ("src/pp.rs", 14),
            ("src/quant.rs", 5),
            ("src/qwen.rs", 7),
            ("src/server.rs", 19),
            ("src/shard_plan.rs", 7),
            ("src/tokenizer.rs", 9),
            ("src/tools.rs", 3),
            ("src/tp.rs", 12),
            ("src/tp_bench.rs", 75),
            ("src/w4a4.rs", 1),
            ("src/wp24.rs", 8),
        ];
        let mut files = Vec::new();
        walk(std::path::Path::new("src"), &mut files);
        assert!(!files.is_empty(), "src walk found nothing - wrong cwd?");
        let mut drift = Vec::new();
        for f in &files {
            let src = std::fs::read_to_string(f).unwrap();
            let got = count_direct_prints(&src);
            let want = baseline.iter().find(|(b, _)| f == b).map(|(_, n)| *n).unwrap_or(0);
            if got != want {
                drift.push(format!("{f}: {got} direct prints, baseline {want} \
  (convert to rprintln!/reprintln! on served paths, or update the table consciously)"));
            }
        }
        assert!(drift.is_empty(), "direct-print baseline drift:\n{}", drift.join("\n"));
    }
}
