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
            // FIX1 #76 (master): +2 = the two [dspark] 1-per-50-step meters (chain + sampled lanes).
            // OUT1: -1 = the graphs-skipped boot line became boot_report::notice(). 35+2-1 = 36.
            ("src/batch.rs", 36),
            ("src/bin/dsv4_attn_replay.rs", 9),
            ("src/bin/dsv4_convert.rs", 3),
            ("src/bin/dsv4_replay.rs", 10),
            ("src/bin/dsv4_stage_debug.rs", 2),
            ("src/bin/vision_cross.rs", 6),
            // S-B9-REL-VISIONFIX: a new dev tool (GPU tower embedding hashes, gate 2), and two more
            // lines of the CPU-tower tool (forward time, embedding hash). Dev tools, not served paths.
            ("src/bin/vision_gpu_hash.rs", 2),
            ("src/bin/w2_preproc_golden.rs", 8),
            ("src/bin/w2_vision_cpu.rs", 10),
            ("src/bin/w2_vision_gpu.rs", 8),
            ("src/bin/w2_vision_load.rs", 6),
            ("src/boot_report.rs", 2),
            // OUT2 (2026-10-16): boot_report.rs = 2 — the dprintln!/deprintln! macro bodies
            // (direct println!/eprintln! behind the --log-level debug check; zero direct prints at normal).
            ("src/cluster.rs", 57), // NAIVE: +1, the one-time `[head] Naive drafter manifest` boot line (ISSUES 98 drafter shipping)
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
            ("src/exl3.rs", 20), // K35 (2026-10-31): master already read 20 vs this row's 18 (drifted, never re-pinned); K35 adds no direct print here: re-pinned consciously // was 18 // P7 B1: the honest load-time width-refusal line (+1)
            ("src/exl3_autotune.rs", 15),
            ("src/exl3_bench.rs", 153), // K8EXP merge (2026-11-07): 152 -> 153 — + the bitwidths section-7 PASS println (probe-only). PFHALF (2026-11-06): 151 -> 152 — the fractional-width (K 1.5/2.5/3.5) PASS println of the moe_pf probe (probe-only, output IS the product). PR #32 port: +1, the EXL3-MOE-PF-HASH line of the synthetic moe_pf probe (probe path). K35 (2026-10-31): master already read 151 vs this row's 149 (PR#27-D3/P7 rows drifted, never re-pinned); K35 removes the fractional-k2 SKIP println (the grouped probe now runs those widths) and adds nothing outside tests: 150, re-pinned consciously // was 149 // PR#27-D3 (2026-10-30): 141 -> 149 - the bitwidths gate prints 8 direct lines (banner, dense/wide/ks per-cell PASS, refusal-predicates PASS, ALL-CELLS verdict, synth8 header+PASS; probe-only, output IS the product) // P7 B10 (2026-10-30): 139 -> 141 — the two explicit-nst smem-cap SKIP lines (per-bits + k4 twin scheds loops; b6/b7 gu-s8 rings over the opt-in cap, the served plan falls back by design; probe-only). P7 B10: 137 -> 139 — the per-width W4 persistent-rows PASS println + the skip line for widths without epi rows (probe-only). P7 B6 (2026-10-30): 136 -> 137 — the synth7 K=7 CHAIN PASS println (probe-only). P7 B5 (2026-10-29): 134 -> 136 — the synth7 leg header + K=7 PASS printlns (probe-only). P7 B3 leg 2 (2026-10-29): 131 -> 134 — the three EPI-ATTR receipt eprintln!s (xh static-smem/opt-in/regs/dev-copy dump; probe-only diagnostic, the 46224>46080 lesson). Earlier: B2 ref-verify + B3 WP20 b5 + B4 MoE-PF bits param prints (+3, probe paths only)
            ("src/exl3_forward.rs", 332), // K35 (2026-10-31): master already read 329 vs this row's 326 (drifted, never re-pinned); K35 adds the three boot-visible once-per-boot eprintln!s (K=x.5 experts ENGAGED decode + PREFILL, the --moe-coop override): 332, re-pinned consciously // was 326 // P7 B9 (2026-10-29): 330 -> 326 — three dprintln! PLE K= lines promoted to boot_report::notice (always-visible width receipt; the notice() call carries no println) + the w4moe plan line folded. P7 B6 (2026-10-30): 329 -> 330 — the PSK width-guard once-per-boot line (boot class, dprintln! inside a OnceLock; the counter is /metrics). P7 B1/B4: width-refusal + pf-predicate boot lines (+2) // P7 merge: 325 -> 327 (df2trace ttr timing lines + df2 head-quantization one-time notices auto-merged from the 1.5bpw branch; conscious update). EXL27C: 324 -> 325 (split-MoE EP deal one-time boot line; conscious update). Earlier: 315 -> 323 (--df2-trace gated diagnostics + Fix5 comment; conscious update). OUT2: 317 -> 315 (the two PLE SSD/RAM decision lines -> boot_report::notice(), one normal line each)
            ("src/exl3_forward/dense.rs", 12),
            ("src/exl3_forward/dhead.rs", 11),
            ("src/exl3_forward/w4s.rs", 8),
            ("src/exl3_forward/wp16.rs", 12),
            ("src/exl3_forward/xtp.rs", 75), // S-B9-REL-INTEG A5: +1, the harness's RANK_IDENTITY FAIL line when pass 1 errors (a probe, not a served path)
            ("src/exl3_kvnll.rs", 10),
            // Q4K: 9 -> 10, the Q4K_HOST line of --probe-exl3-kvq (the q4k host reference's own
            // verdict line: probe-only, one-shot, the same class as the KVQ_SELFTEST lines above;
            // conscious update).
            // S-B9-REL-PPL: the one-shot `--perplexity` EXL3 tool prints its identity / per-window /
            // reference / RESULT lines directly (a CLI measurement tool, not a served path; the order
            // and immediacy of its output is the product, like exl3_kvnll.rs above).
            // S-B9-REL-PPL2 W5: 9 -> 10, the PPL_BOUNDS line (the bounds a run was judged by, printed
            // next to PPL_STATS before the checks; the same one-shot tool-output class).
            // S-B9-REL-PPL2 1b: 10 -> 4. The per-window lines, the PPL_HEAD_CHECK / PPL_STATS /
            // PPL_BOUNDS lines and RESULT are written through a `&mut dyn Write` handed to `score_all` /
            // `finish_run` (stdout in the tool, a buffer in the unit tests that prove a failed run prints no
            // RESULT line); what is left direct is the header, PPL_IDENT and the decode-reference lines.
            ("src/exl3_ppl.rs", 4),
            ("src/exl3_regime.rs", 1), // S-B9-REL-QSAREGIME: the probe prints through one closure (not a served path)
            // MTPREPROBE2 W1.5: +2 = the two step_plain a-xcheck record-failure branches
        // (eprintln, matching the existing sibling site in finish()/xcheck drain).
        // S-B9-REL-SMALLFIXES item 1: 87 -> 86, the two MTP boot-banner println! became one
        // (the line is built by the testable mtp_banner()).
        ("src/exl3_serve.rs", 106), // MTPAGREE1 (v0.7.7): 103 -> 106, the [df2-tp2] lane dump on the lockstep-failure arm, the [round-trace] line behind --round-trace and the [tp-ident] differing-digest line (failure / diagnostic paths; TOPKM1 hygiene, conscious update). Earlier: EXL27C: 88 -> 97 (--df2-trace gated dispatch/round prints; conscious update). +1 AGREE16: the width drill's line (test drill; one-shot since AGREE16B, right before a link abort)
            ("src/exl3_tune.rs", 7),
            ("src/exl3_wp27.rs", 8),
            ("src/gptq.rs", 42),
            ("src/gpu.rs", 323),
            ("src/json_schema.rs", 1),
            ("src/kernels.rs", 3),
            ("src/logq.rs", 3),
            // +3 (S-B9-REL-PRPORT4): the --prefill-interleave lines of --help (one-time banner class);
            // +1 (S-B9-REL-QSAREGIME): the PROBE_REGIME_FAIL line of the --probe-exl3-regime dispatch (probe exit path)
            // +7 (S-B9-REL-PPL): 5 --help lines naming the EXL3 --perplexity flags (one-time help class)
            // + 2 PERPLEXITY_EXL3_FAIL exit-path lines of the --perplexity EXL3 dispatch (probe exit path)
            // S-B9-REL-SURFACE2 item 5: the hand-written --mtp (4 lines) and --prefill-interleave
            // (3 lines) entries left print_help (each option is listed once, by its registry
            // line) and the SPECULATION header went from 1 line to 2: -6.
            // Both together: 1556 + 7 - 6 = 1557; -2: the hand-written --vision-cpu help lines left print_help
            // (VISIONFIX2 made it a registry flag: listed once, by its registry line) = 1555.
            // S-B9-REL-PPL2 W4: 1557 -> 1552: -4 and -3 hand-written --perplexity help lines (the mode line
            // in MODES and the entry in the second block left print_help: the tool's options are registry
            // rows now and are listed once, under PERPLEXITY TOOL), +1 pointer line in MODES, +1 the
            // legacy path's `--window < 2` refusal (a CLI exit path, like PERPLEXITY_EXL3_FAIL).
             // F9b: +1 the bare-tp info line (stderr, one line, once) // S-B9-REL-INTEG A1: +2 --help lines (the two TP drill flags, one line each)
            ("src/memwatch.rs", 3),
            ("src/model.rs", 3),
            ("src/net.rs", 14), // +1 AGREE16: the width-refusal line of agree_ext (printed once, right before the link abort)
            // S-B9-REL-FINALFIX F7 (2026-10-11): opts.rs = 2 direct prints.
            // 1 = the FATAL env-var rejection line at boot (served, pre-router, by design);
            // +1 = the #[ignore]d f7_differential long-run progress line (test-only).
            ("src/opts.rs", 3), // F9: +1 the f9_differential count line, test code read with --nocapture
            ("src/pp.rs", 14),
            ("src/quant.rs", 5),
            ("src/qwen.rs", 7),
            // S-B9-REL-BENCHFIX2: +2 = the "[serve] sse keep-alive: ..." boot line (println!, the
            // one-time banner class, printed by http_listen before the socket exists) and the
            // FATAL line of fatal_option (direct by the H7 rule above: printed, then flush_and_exit)
            ("src/server.rs", 21),
            ("src/shard_plan.rs", 9), // FP4L (2026-10-29): +1, the `fp4l shard opens` line of the --probe-shard-plan FP4L arm (probe path). NAIVE: +1, the `owner mismatch` line of the Naive arm of --probe-shard-plan (probe path)
            ("src/exl3_forward/fp4l_fwd.rs", 13), // FP4L: the --probe-fp4l-gpu verdict lines (probe path; the served FP4L paths log via rprintln!/drprintln!); +1 (2026-10-30) FP4L_GPU_PAIR
            ("src/fp4l/probe.rs", 29), // FP4L-0 (2026-10-29): the --probe-fp4l CPU receipts print to stdout (probe verdict lines; never a served path); +2 (2026-10-30): the FP4L_PROBE_FAIL / FP4L_GPU_FAIL exit lines of the probe dispatch (moved out of main.rs).
            // BASICS (owner 2026-11-02): the version line (stdout, or stderr under --print-config) and the
            // configuration block: boot banners, which stay direct by this module's rule.
            ("src/startup.rs", 3),
            ("src/tokenizer.rs", 9),
            // S-B9-REL-SURFACE2 item 10: + 2, the ignored `tool_quote_probe` instrument (test code).
            // S-B9-REL-SURFACE2B R1: + 5, the count tables of `exhaustive_small_blocks` and the
            // timing line of the 1 MB test (test code, read with --nocapture).
            // S-B9-TOOLPARSE3: + 4 in tools.rs (the megabyte-block and shadow-cost instrument lines, test
            // code, read with --nocapture), and src/tools/cost.rs: 32, the measurement tables of the
            // cost model (ignored instruments and one pinned test, test code only).
            ("src/tools.rs", 15),
            ("src/tools/cost.rs", 32),
            // S-B9-TOOLPARSE3 part B: the in-tree default-identity test and the frozen v0.7.3 parser it
            // compares with (test builds only; the frozen copy keeps the original's two alarm prints).
            ("src/tools/default_identity.rs", 3),
            ("src/tools/v073_frozen.rs", 2),
            // rel/prport3 (review R8): the tp.rs bring-up evidence lines now go through
            // reprintln! (they fire before the log router exists; reprintln! prints directly
            // until one does), so tp.rs returns to 12. main.rs: the --vision-fail-gemm help
            // block moved into opts_table (Cat::Diag -> --help-diag), 1554 -> 1552.
            // CLIPRUNE1: main.rs +1, the --help-advanced dispatch print (same shape as the
            // --help one). Conscious update, not a conversion miss.
            // OUT1 (2026-10-12): 1555 -> 1547 — the ngram/tree/lanes EXPERIMENTAL boot lines became
            // boot_report::notice(), the DSV4 sharded refusal became boot_report::error(), the two
            // [vision] degradations became boot_report::warn(), the ctx clamp became
            // boot_report::notice(); the boot_report vocabulary itself routes via logq::send.
            // OUT1 DSV4 leg: 1547 -> 1546 — the [head] deepseek_v4 detection line became boot_report::notice()
            // (the table kept 1547: a later +1 landed before this baseline was re-pinned).
            // MAXBATCH1 (2026-10-21): 1547 -> 1551 — the four --max-batch EXL3 cap/clamp help
            // lines (truthful help, same println! shape as --max-tokens').
            // DFLASH2-GEN 2026-10-23: 1552 -> 1551: the two-line --gen-dflash2-synth help
            // entry merged into one line (geometry-dynamic wording; not a served path).
            // DFLASH2-GEN 2026-10-23: 1551 -> 1555 — the three --df2-mask-override help lines
            // and the disabled-at-CLI boot notice (arg-parse path, not a served path).
            // NAIVEPORT (the port merge): naive's +33 lands on top of master's own +15 (the MAXBATCH1 / DFLASH2-GEN
            // help lines above); re-pinned to the count of the MERGED tree (1547 + 33 + 15 = 1595, counted with this
            // module's counter on the merged main.rs before the build). The row below is OPERATIVE (find() takes the
            // first); naive-n05's original row is kept verbatim after it for the port's line-identity check.
            ("src/main.rs", 1637), // SELFUPD: 1632 -> 1637 (--update/--no-update-check help + notices)
        ("src/self_update.rs", 23), // SELFUPD: updater is an offline-exit path (no serving), direct prints intended
            // NAIVE history of src/main.rs: +33, the --probe-naive-* dispatches' NAIVE_*_FAIL exit-path lines, their usage expect()s' siblings, and the head/node Naive TP boot lines (probe/boot classes)
            // Conscious updates, not conversion misses.
        ("src/tp.rs", 13),
            ("src/tp_bench.rs", 75),
            ("src/w4a4.rs", 1),
            ("src/wp24.rs", 8),
            // NAIVE (the Naive-N0.5-Flash family tree, src/naive/*, ported from the naive-n05 line 2026-10-21): the family's
            // own probes and its TP serve loop print directly (probe verdict lines, one-time boot lines, the `[naive-serve]`
            // request log). Counted consciously at the port; ISSUES (naive-n05 numbering) #105 tracks the conversion to
            // rprintln!/reprintln! on the served paths. Files of the tree not listed here print nothing.
            ("src/naive/cpu.rs", 8),
            ("src/naive/decode.rs", 24),
            ("src/naive/dequant.rs", 4),
            ("src/naive/drafter_gpu.rs", 12),
            ("src/naive/frac.rs", 2),
            ("src/naive/gpu.rs", 23),
            ("src/naive/model.rs", 33), // LOADFAST (f2571038): the per-layer routed up_proj scale println! removed, the layer-loaded line is a dprintln! (34 -> 33; TOPKM1 hygiene, conscious update)
            ("src/naive/pack.rs", 11),
            ("src/naive/serve.rs", 68),
            ("src/naive/shard.rs", 3),
            ("src/naive/stage.rs", 10), // LOADFAST (f2571038): the staged-layer loader's cache/stage progress + warning lines, shipped in v0.7.6 without a baseline row (TOPKM1 hygiene; conscious update)
            ("src/naive/tp.rs", 7),
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
