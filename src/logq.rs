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

use std::sync::atomic::{AtomicIsize, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender, TrySendError, Receiver};
use std::sync::OnceLock;
use std::time::Duration;

/// Queue capacity: lines. ~100 KiB worst case; a healthy reader drains this in milliseconds.
pub const CAP: usize = 4096;

static DROPPED: AtomicU64 = AtomicU64::new(0);
static QUEUED: AtomicIsize = AtomicIsize::new(0);
static TX: OnceLock<SyncSender<Line>> = OnceLock::new();

/// ANSI color policy (`--color auto|always|never`, default `auto` = TTY-gated, zero env vars per §7).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ColorMode { Auto = 0, Always = 1, Never = 2 }

static COLOR_MODE: AtomicU8 = AtomicU8::new(ColorMode::Auto as u8);

pub fn set_color_mode(m: ColorMode) { COLOR_MODE.store(m as u8, Ordering::Relaxed); }

pub fn color_mode() -> ColorMode {
    match COLOR_MODE.load(Ordering::Relaxed) {
        1 => ColorMode::Always,
        2 => ColorMode::Never,
        _ => ColorMode::Auto,
    }
}

fn parse_color_mode(v: &str) -> ColorMode {
    match v.trim() {
        "always" | "on" => ColorMode::Always,
        "never" | "off" => ColorMode::Never,
        _ => ColorMode::Auto,
    }
}

/// Early scan of argv before `opts::parse_cli` so line 1 (banner) already honors `--color`.
pub fn set_color_from_args(args: &[String]) {
    if let Some(v) = crate::arg_value(args, "--color") {
        set_color_mode(parse_color_mode(v));
    }
}

/// Refresh `--color` from the parsed options store (also called on TP nodes after `install_head`).
pub fn refresh_color_from_opts() {
    if let Ok(v) = crate::opts::var(crate::opt!("color")) {
        set_color_mode(parse_color_mode(&v));
    }
}

/// Whether ANSI color should be emitted on stderr (`err = true`) or stdout (`err = false`).
pub fn color_enabled(err: bool) -> bool {
    match color_mode() {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => unsafe { libc::isatty(if err { 2 } else { 1 }) != 0 },
    }
}

/// Current local wall-clock datetime (`YYYY-MM-DD HH:MM:SS`) placed after each message's `[tag]`.
pub fn now_datetime() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Split a leading `[tag]` (e.g. `[boot]`, `[req]`, `[exl3-serve]`, `[dsv41]`) from the rest of the line.
/// Rejects brackets containing spaces, commas, or non-tag payloads like `[1, 2, 3]`.
pub fn split_bracket_tag(line: &str) -> Option<(&str, &str)> {
    let bytes = line.as_bytes();
    if bytes.first().copied() != Some(b'[') { return None; }
    let limit = bytes.len().min(32);
    let close = bytes[1..limit].iter().position(|&b| b == b']')? + 1;
    if close <= 1 { return None; }
    if !bytes[1..close].iter().all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':' | b'/')) {
        return None;
    }
    let tag = &line[..=close];
    let rest = &line[close + 1..];
    let rest = rest.strip_prefix(' ').unwrap_or(rest);
    Some((tag, rest))
}

fn ansi(code: &str, s: &str) -> String { format!("\x1b[{code}m{s}\x1b[0m") }

fn is_error_body(rest: &str) -> bool {
    rest.starts_with("error:") || rest.starts_with("ERROR") || rest.starts_with("FATAL")
        || rest.starts_with("ABORT") || rest.contains(" failed after ") || rest.contains(" FAILED")
}

fn is_warn_body(rest: &str) -> bool {
    rest.starts_with("warn:") || rest.starts_with("WARNING") || rest.starts_with("WARN ")
}

fn style_req_body(rest: &str) -> String {
    if rest.starts_with("400 ") || rest.starts_with("500 ") || rest.starts_with("503 ") {
        return ansi("33", rest);
    }
    let mut out = String::with_capacity(rest.len() + 32);
    for (idx, tok) in rest.split(' ').enumerate() {
        if idx > 0 { out.push(' '); }
        if tok.ends_with("tok/s") || tok.ends_with("tok/s)") {
            out.push_str(&ansi("1;32", tok));
        } else if let Some(v) = tok.strip_prefix("ttft=") {
            out.push_str(&ansi("2", "ttft="));
            out.push_str(&ansi("36", v));
        } else if let Some(v) = tok.strip_prefix("cached=") {
            out.push_str(&ansi("2", "cached="));
            if v != "0" { out.push_str(&ansi("32", v)); } else { out.push_str(v); }
        } else if let Some((k, v)) = tok.split_once('=') {
            out.push_str(&ansi("2", &format!("{k}=")));
            out.push_str(v);
        } else {
            out.push_str(tok);
        }
    }
    out
}

fn style_boot_body(rest: &str) -> String {
    if let Some(after) = rest.strip_prefix("stage ") {
        if let Some(done_idx) = after.find(" done ") {
            let name = &after[..done_idx];
            let tail = &after[done_idx + 1..];
            return format!("stage {} {}", ansi("1", name), ansi("32", tail));
        }
        if let Some(name) = after.strip_suffix(" begin") {
            return format!("stage {} {}", ansi("1", name), ansi("2", "begin"));
        }
    }
    // Progress lines: `<what>:  45% (...)` or `100% (...)`
    if let Some(pct_idx) = rest.find('%') {
        if pct_idx >= 1 && rest[..pct_idx].ends_with(|c: char| c.is_ascii_digit()) {
            let num_start = rest[..pct_idx].rfind(|c: char| !c.is_ascii_digit() && c != ' ').map_or(0, |i| i + 1);
            let pct_slice = &rest[num_start..=pct_idx];
            let code = if pct_slice.trim() == "100%" { "1;32" } else { "1;36" };
            return format!("{}{}{}", &rest[..num_start], ansi(code, pct_slice), &rest[pct_idx + 1..]);
        }
    }
    rest.to_string()
}

fn style_serving_line(first: &str) -> String {
    let mut out = String::with_capacity(first.len() + 64);
    for (idx, tok) in first.split(' ').enumerate() {
        if idx > 0 { out.push(' '); }
        if idx == 0 && tok == "SERVING" {
            out.push_str(&ansi("1;32", "SERVING"));
        } else if let Some((k, v)) = tok.split_once('=') {
            out.push_str(&ansi("2", &format!("{k}=")));
            match k {
                "model" | "url" => out.push_str(&ansi("1;36", v)),
                "boot" => out.push_str(&ansi("32", v)),
                "tp" | "family" => out.push_str(&ansi("1", v)),
                _ => out.push_str(v),
            }
        } else {
            out.push_str(tok);
        }
    }
    out
}

/// Pure formatter: inserts `ts` right after a leading `[tag]` and (when `color` is true) applies
/// ANSI colors by severity and subsystem. Leaves untagged lines unchanged in plain mode.
pub fn format_log_line(line: &str, ts: &str, color: bool) -> String {
    let (first, tail) = match line.split_once('\n') {
        Some((f, t)) => (f, Some(t)),
        None => (line, None),
    };
    let head = if let Some((tag, rest)) = split_bracket_tag(first) {
        if !color {
            if rest.is_empty() { format!("{tag} {ts}") } else { format!("{tag} {ts} {rest}") }
        } else {
            let ts_styled = ansi("2", ts);
            let (tag_code, body) = if is_error_body(rest) {
                if let Some(after) = rest.strip_prefix("error:") {
                    ("1;31", format!("{}{}", ansi("1;31", "error:"), ansi("31", after)))
                } else {
                    ("1;31", ansi("31", rest))
                }
            } else if is_warn_body(rest) {
                if let Some(after) = rest.strip_prefix("warn:") {
                    ("1;33", format!("{}{}", ansi("1;33", "warn:"), ansi("33", after)))
                } else {
                    ("1;33", ansi("33", rest))
                }
            } else if let Some(after) = rest.strip_prefix("notice:") {
                ("1;33", format!("{}{}", ansi("33", "notice:"), after))
            } else if tag == "[req]" {
                let code = if rest.starts_with("400 ") || rest.starts_with("500 ") || rest.starts_with("503 ") { "1;33" } else { "1;32" };
                (code, style_req_body(rest))
            } else if tag == "[boot]" {
                ("1;36", style_boot_body(rest))
            } else if matches!(tag, "[tp]" | "[exl3-tp]" | "[head]" | "[node]" | "[net_shim]") {
                ("1;35", rest.to_string())
            } else if matches!(tag, "[thermal]" | "[gpu-wait]" | "[kv-spill]" | "[kv-vmm]" | "[opts]" | "[logq]") {
                ("1;33", rest.to_string())
            } else {
                ("1;36", rest.to_string())
            };
            let tag_styled = ansi(tag_code, tag);
            if body.is_empty() {
                format!("{tag_styled} {ts_styled}")
            } else {
                format!("{tag_styled} {ts_styled} {body}")
            }
        }
    } else if !color {
        first.to_string()
    } else if first.starts_with("SERVING ") {
        style_serving_line(first)
    } else if first.starts_with("FATAL:") {
        ansi("1;31", first)
    } else {
        first.to_string()
    };
    match tail {
        Some(t) => format!("{head}\n{t}"),
        None => head,
    }
}

/// Format a log line for stdout (`err = false`) or stderr (`err = true`) with local datetime + policy-gated ANSI color.
pub fn format_for_stream(err: bool, text: &str) -> String {
    format_log_line(text, &now_datetime(), color_enabled(err))
}

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
            let formatted = format_for_stream(l.err, &l.text);
            if l.err { eprintln!("{formatted}"); } else { println!("{formatted}"); }
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
    let formatted = format_for_stream(l.err, &l.text);
    if l.err {
        let mut h = std::io::stderr().lock();
        let _ = h.write_all(formatted.as_bytes()).and_then(|_| h.write_all(b"\n"));
    } else {
        let mut h = std::io::stdout().lock();
        let _ = h.write_all(formatted.as_bytes()).and_then(|_| h.write_all(b"\n"));
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
                        "{}",
                        format_for_stream(
                            true,
                            &format!("[logq] warn: recovered: {n} log lines were dropped while the log reader was stalled (velogb10_log_lines_dropped_total)")
                        )
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

    #[test]
    fn format_log_line_inserts_datetime_after_tag_and_styles_with_ansi() {
        let ts = "2026-11-24 20:28:35";
        // Plain mode: datetime goes right after [tag], zero ANSI escapes
        assert_eq!(
            format_log_line("[boot] stage weights begin", ts, false),
            "[boot] 2026-11-24 20:28:35 stage weights begin"
        );
        assert_eq!(
            format_log_line("[exl3-serve] slot=0 ok", ts, false),
            "[exl3-serve] 2026-11-24 20:28:35 slot=0 ok"
        );
        // Untagged lines stay untouched in plain mode (SERVING, arrays like [1, 2])
        assert_eq!(format_log_line("SERVING family=dsv4 model=x", ts, false), "SERVING family=dsv4 model=x");
        assert_eq!(format_log_line("[1, 2, 3]", ts, false), "[1, 2, 3]");
        // Multi-line preserves continuation lines untouched
        assert_eq!(
            format_log_line("[boot] warn: line 1\n  line 2", ts, false),
            "[boot] 2026-11-24 20:28:35 warn: line 1\n  line 2"
        );
        // Color mode: ANSI codes applied, datetime dimmed, tags & severities colored
        let err = format_log_line("[boot] error: bad path", ts, true);
        assert!(err.contains("\x1b[1;31m[boot]\x1b[0m \x1b[2m2026-11-24 20:28:35\x1b[0m"));
        assert!(err.contains("\x1b[1;31merror:\x1b[0m"));
        let done = format_log_line("[boot] stage weights done 3.2s", ts, true);
        assert!(done.contains("\x1b[1;36m[boot]\x1b[0m \x1b[2m2026-11-24 20:28:35\x1b[0m"));
        assert!(done.contains("\x1b[32mdone 3.2s\x1b[0m"));
        let req = format_log_line("[req] prompt=100 cached=50 gen=20 ttft=42ms 87.6tok/s", ts, true);
        assert!(req.contains("\x1b[1;32m[req]\x1b[0m \x1b[2m2026-11-24 20:28:35\x1b[0m"));
        assert!(req.contains("\x1b[1;32m87.6tok/s\x1b[0m"));
        let srv = format_log_line("SERVING family=dsv4 model=DSV4 url=http://0.0.0.0:9000 boot=61.0s", ts, true);
        assert!(srv.starts_with("\x1b[1;32mSERVING\x1b[0m "));
        assert!(srv.contains("\x1b[1;36mDSV4\x1b[0m"));
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
            // (src/batch.rs split into src/batch/*.rs by SPLIT1; the 36 prints are spread over the rows below, mod.rs has none.)
            ("src/batch/admission.rs", 3),
            ("src/batch/df2.rs", 6),
            ("src/batch/dspark.rs", 10),
            ("src/batch/lifecycle.rs", 13),
            ("src/batch/mtp_lane.rs", 1),
            ("src/batch/mtp_policy.rs", 3),
            ("src/bin/dsv4_attn_replay.rs", 9),
            ("src/bin/dsv4_convert.rs", 3),
            ("src/bin/dsv4_replay.rs", 10),
            ("src/bin/dsv4_stage_debug.rs", 2),
            ("src/bin/chat_template_dump.rs", 4), // INKLING2 T1: a dev tool (chat-template dump), not a served path (conscious update)
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
            // SPLIT1 cluster.rs (57) -> per-file rows; the sum is 58 because pack_manifest_tests.rs (a moved test module, no cfg(test) wrapper in its own file) is counted raw (+1 print that sat inside the stripped module)
            ("src/cluster/head.rs", 21), // VELO-FABRIC: +2 one-time boot deprintln! lines for RoCEv2 bulk RDMA active/fallback during weight sync. NAIVE: +1, the one-time `[head] Naive drafter manifest` boot line (ISSUES 98 drafter shipping)
            ("src/cluster/models.rs", 13),
            ("src/cluster/node.rs", 15), // VELO-FABRIC: +2 one-time boot deprintln! lines for RoCEv2 bulk RDMA active/fallback during weight sync. LS2: +1, the one-time `[node] N Engram head file(s) already staged` line at node boot (sync phase, once per session; the other node boot lines are direct too)
            ("src/cluster/mod.rs", 9),
            ("src/cluster/discovery.rs", 4),
            ("src/cluster/pack_manifest_tests.rs", 1),
            ("src/copy_span/tests.rs", 2), // SPLIT1 copy_span: no row before (0 outside cfg(test)); the moved test file (no cfg(test) wrapper in its own file) is counted raw: +2
            ("src/dflash.rs", 2),
            // SPLIT1: src/dflash2/round.rs split into src/dflash2/round/*.rs; the 7 prints are spread over the rows below.
            ("src/dflash2/round/draft.rs", 6),
            ("src/dflash2/round/load.rs", 1),
            ("src/dflash2/stepdump.rs", 4),
            ("src/dspark/round.rs", 4),
            // SPLIT1: src/dsv4_attn.rs split into src/dsv4_attn/*.rs; the 1 prints are spread over the rows below.
            ("src/dsv4_attn/attn.rs", 1),
            ("src/dsv4_convert.rs", 9),
            ("src/dsv4_dspark.rs", 7),
            ("src/dsv4_graph.rs", 5),
            // SPLIT1: src/dsv4_model.rs split into src/dsv4_model/*.rs; the 13 prints are spread over the rows below.
            ("src/dsv4_model/forward.rs", 6),
            // DSV41 INT1 (2026-11-18), conscious baseline rows for the merged I0 / ENGa / ENGb / ENC files (none is a served path):
            ("src/dsv41/engram/pack.rs", 2), // ENGb --dsv41-engram-pack: per-head result line + "pack done" line (offline CLI tool, no GPU, no model load)
            ("src/dsv41/engram/verify.rs", 4), // ENGb --dsv41-engram-verify: SKIP / sha256 / sampled-rows / PASS-FAIL verdict lines (offline CLI tool)
            ("src/dsv41/engram/iobench.rs", 8), // ENGb --dsv41-engram-iobench: the markdown result table the benchmark exists to print (offline CLI tool)
            ("src/dsv41/engram/golden_tests.rs", 8), // ENGa: a whole #[cfg(test)] file (declared `#[cfg(test)] mod golden_tests;`, so the counter sees no wrapper): test SKIP / receipt eprintln!s
            ("src/dsv41/chat/tests.rs", 4), // ENC: a whole #[cfg(test)] file: test SKIP / NOTE / receipt eprintln!s
            ("src/dsv41/cli.rs", 26), // DSV41 I0: the --dsv41-* / --probe-dsv41-* tool entry points (result lines and FAIL lines); no served path. 22 -> 26: the KN1 fp8, KN2 mxfp4, KC kv, KE and KI indexer probe-result print loops (+4 net of lines the merges folded together)
            ("src/dsv41/weights/exl3/probe.rs", 2), // KE --probe-dsv41-kernel --dsv41-kernel-set trellis|plain: the result lines and DSV41_KE_OK (probe, no served path)
            ("src/dsv41/serve/tests.rs", 1), // SV: whole #[cfg(test)] file: the SKIP eprintln! when the lab fixture is absent
            ("src/dsv41/serve/engine.rs", 2), // SV: the FATAL banner when the lane loop panics + (SE) the one-line error when the forward's idle call fails; error/FATAL lines stay direct by the logq policy (the 503 state follows)
            ("src/dsv41/gpu/mix_tests.rs", 15), // KM: whole #[cfg(test)] file: receipt lines of the GPU-vs-f64 mixer tests
            ("src/dsv41/gpu/pp1_tests.rs", 3), // DSV41 PP1: whole #[cfg(test)] file: the GPU bit-identity / timing receipt lines
            ("src/dsv41/gpu/pa1_tests.rs", 3), // PA1: whole #[cfg(test)] file: receipt lines of the GPU bit-identity tests
            ("src/dsv41/engram/io_probe.rs", 22), // EIO --probe-dsv41-engram-io: the result / latency tables the probe exists to print (probe, no served path)
            ("src/dsv41/load/cli.rs", 6), // LS --dsv41-shard-plan / --dsv41-ship tools: plan markdown, DSV41_SHARD_PLAN_OK, rank summary, sync + DSV41_SHIP_OK (offline tools). 5 -> 6: LS2 --dsv41-stage (the DSV41_STAGE_OK result line; offline tool)
            ("src/dsv41/serve/mod.rs", 4), // VELO-FABRIC: +1 one-time boot dprintln! loop printing metrics::boot_receipt_lines() before serving. (SE2: +1 = the one-shot boot line describing --tool-scrub when it is off, dprintln!, before any request) (SE: +1 = the fatal "dsv41 http server stopped" line before exit(1), direct by the FATAL policy) LS2: the one-shot boot line `memory fit: ...` (dprintln!, once at serve boot, before any request; the cap notice itself goes through boot_report::notice)
            ("src/dsv41/cpu/attn.rs", 2), // OA --probe-dsv41-oracle --dsv41-oracle-part attn: the per-layer result lines and DSV41_ORACLE_ATTN_OK (offline oracle tool, no served path)
            ("src/dsv41/cpu/dspark.rs", 9), // OD --dsv41-oracle-part dspark: the header, per-step trace, markdown result table and DSV41_ORACLE_DSPARK_OK (offline oracle tool; the test-module eprintln!s are not counted)
            ("src/dsv41/ced/mod.rs", 1), // DSV41 CE --probe-dsv41-ced: `say` prints each gate line as it is produced and keeps it (probe-only; no served path)
            ("src/dsv41/gpu/dspark/gate.rs", 3), // DSV41 DS --probe-dsv41-dspark: `say`, the per-fixture result lines and the sampler real-rows line (probe-only; no served path)
            ("src/dsv41/tp/serve.rs", 2), // SE: the FATAL banner when a TP op fails on the head (ranks can no longer be trusted; the server answers 503); FATAL stays direct by the logq policy. MEM1: +1 dprintln! (the per-rank memory ledger breakdown at boot, --log-level debug only)
            ("src/dsv41/tp/launch.rs", 17), // DSV41 TPC/TP --probe-dsv41-tp smoke + model-program head/node launch: boot receipts, per-line program output and DSV41_TP_MODEL_OK/FAIL (probe-only). 13 -> 17 (SE): serve_head / serve_node boot receipts (the [head] pack and node-sync lines, the [node] rank receipt, once at boot; the per-round lines are the callback's, not counted)
            ("src/dsv41/weights/exl3/q1_oracle.rs", 10), // DSV41 Q1 --probe-dsv41-q1: per-class result lines and DSV41_Q1_ORACLE_OK (offline decode oracle, no served path)
            ("src/dsv41/weights/partials_probe.rs", 1), // DSV41 PK3 partials probe: `say` prints each result line as it is produced and keeps it (probe-only; no served path)
            ("src/dsv41/fwd/mod.rs", 1), // DSV41 SL --probe-dsv41-slice-model: the gate's result lines, printed as they are produced (probe-only; no served path)
            ("src/dsv4_model/load.rs", 7),
            ("src/engine.rs", 1),
            ("src/exl3.rs", 20), // K35 (2026-10-31): master already read 20 vs this row's 18 (drifted, never re-pinned); K35 adds no direct print here: re-pinned consciously // was 18 // P7 B1: the honest load-time width-refusal line (+1)
            ("src/exl3_autotune/cli_main.rs", 5), // SPLIT1 exl3_autotune: the old row "src/exl3_autotune.rs" (15 outside cfg(test)) is split per file; the moved test file (no cfg(test) wrapper in its own file) is counted raw: +1, so the rows sum to 16
            ("src/exl3_autotune/session.rs", 1),
            ("src/exl3_autotune/tests.rs", 1),
            ("src/exl3_autotune/tune_cli.rs", 8),
            ("src/exl3_autotune/tuner_plan.rs", 1),
            // SPLIT1 exl3_bench: the old row "src/exl3_bench.rs" (153 outside cfg(test)) is split per file below; the moved test files (no cfg(test) wrapper in their own file) are counted raw: +9 = 2+1+1+3 (k23/k35 goldens) +2 (wp20_suh_tests). Old row history: // K8EXP merge (2026-11-07): 152 -> 153 — + the bitwidths section-7 PASS println (probe-only). PFHALF (2026-11-06): 151 -> 152 — the fractional-width (K 1.5/2.5/3.5) PASS println of the moe_pf probe (probe-only, output IS the product). PR #32 port: +1, the EXL3-MOE-PF-HASH line of the synthetic moe_pf probe (probe path). K35 (2026-10-31): master already read 151 vs this row's 149 (PR#27-D3/P7 rows drifted, never re-pinned); K35 removes the fractional-k2 SKIP println (the grouped probe now runs those widths) and adds nothing outside tests: 150, re-pinned consciously // was 149 // PR#27-D3 (2026-10-30): 141 -> 149 - the bitwidths gate prints 8 direct lines (banner, dense/wide/ks per-cell PASS, refusal-predicates PASS, ALL-CELLS verdict, synth8 header+PASS; probe-only, output IS the product) // P7 B10 (2026-10-30): 139 -> 141 — the two explicit-nst smem-cap SKIP lines (per-bits + k4 twin scheds loops; b6/b7 gu-s8 rings over the opt-in cap, the served plan falls back by design; probe-only). P7 B10: 137 -> 139 — the per-width W4 persistent-rows PASS println + the skip line for widths without epi rows (probe-only). P7 B6 (2026-10-30): 136 -> 137 — the synth7 K=7 CHAIN PASS println (probe-only). P7 B5 (2026-10-29): 134 -> 136 — the synth7 leg header + K=7 PASS printlns (probe-only). P7 B3 leg 2 (2026-10-29): 131 -> 134 — the three EPI-ATTR receipt eprintln!s (xh static-smem/opt-in/regs/dev-copy dump; probe-only diagnostic, the 46224>46080 lesson). Earlier: B2 ref-verify + B3 WP20 b5 + B4 MoE-PF bits param prints (+3, probe paths only)
            ("src/exl3_bench/attn.rs", 8),
            ("src/exl3_bench/bitwidths.rs", 7),
            ("src/exl3_bench/dense.rs", 23),
            ("src/exl3_bench/gemm.rs", 21),
            ("src/exl3_bench/hc.rs", 13),
            ("src/exl3_bench/k23_kernel_wheel_golden.rs", 1),
            ("src/exl3_bench/k23_serving_gemm_layout_golden.rs", 3),
            ("src/exl3_bench/k23_serving_gemm_wheel_golden.rs", 2),
            ("src/exl3_bench/k23_serving_real_call.rs", 2),
            ("src/exl3_bench/k35_kernel_wheel_golden.rs", 3),
            ("src/exl3_bench/moe_epi.rs", 11),
            ("src/exl3_bench/moe_epi_k4.rs", 4),
            ("src/exl3_bench/moe_pf.rs", 6),
            ("src/exl3_bench/router.rs", 16),
            ("src/exl3_bench/sampler.rs", 9),
            ("src/exl3_bench/select.rs", 6),
            ("src/exl3_bench/tp4.rs", 12),
            ("src/exl3_bench/wide.rs", 17),
            ("src/exl3_bench/wp20_suh_tests.rs", 2),
            ("src/exl3_forward/mod.rs", 0), // SPLIT1 exl3_forward: the old single-file row (332 outside cfg(test)) is split per file below (mod.rs itself prints nothing; the moved test file flash256_stage_tests is counted raw: +2, so the rows sum to 334). Old row history: K35 (2026-10-31): master already read 329 vs this row's 326 (drifted, never re-pinned); K35 adds the three boot-visible once-per-boot eprintln!s (K=x.5 experts ENGAGED decode + PREFILL, the --moe-coop override): 332, re-pinned consciously // was 326 // P7 B9 (2026-10-29): 330 -> 326 — three dprintln! PLE K= lines promoted to boot_report::notice (always-visible width receipt; the notice() call carries no println) + the w4moe plan line folded. P7 B6 (2026-10-30): 329 -> 330 — the PSK width-guard once-per-boot line (boot class, dprintln! inside a OnceLock; the counter is /metrics). P7 B1/B4: width-refusal + pf-predicate boot lines (+2) // P7 merge: 325 -> 327 (df2trace ttr timing lines + df2 head-quantization one-time notices auto-merged from the 1.5bpw branch; conscious update). EXL27C: 324 -> 325 (split-MoE EP deal one-time boot line; conscious update). Earlier: 315 -> 323 (--df2-trace gated diagnostics + Fix5 comment; conscious update). OUT2: 317 -> 315 (the two PLE SSD/RAM decision lines -> boot_report::notice(), one normal line each)
            ("src/exl3_forward/attn.rs", 6),
            ("src/exl3_forward/attn_gates.rs", 3),
            ("src/exl3_forward/bench.rs", 42),
            ("src/exl3_forward/consts.rs", 2),
            ("src/exl3_forward/df2.rs", 10),
            ("src/exl3_forward/draft.rs", 2),
            ("src/exl3_forward/forward_step.rs", 1),
            ("src/exl3_forward/hc.rs", 4),
            ("src/exl3_forward/ladder.rs", 1),
            ("src/exl3_forward/lmh.rs", 8),
            ("src/exl3_forward/load.rs", 28),
            ("src/exl3_forward/moe.rs", 8),
            ("src/exl3_forward/moe_wp20.rs", 6),
            ("src/exl3_forward/mpk.rs", 5),
            ("src/exl3_forward/mtp_tap.rs", 3),
            ("src/exl3_forward/ple.rs", 5),
            ("src/exl3_forward/prefill.rs", 5),
            ("src/exl3_forward/prefill_attn.rs", 4),
            ("src/exl3_forward/prefill_hc.rs", 4),
            ("src/exl3_forward/prefill_moe.rs", 7),
            ("src/exl3_forward/prefill_warmup.rs", 10),
            ("src/exl3_forward/probes_lt.rs", 12),
            ("src/exl3_forward/probes_pack23.rs", 31),
            ("src/exl3_forward/probes_packstate.rs", 32),
            ("src/exl3_forward/probes_prefillx.rs", 31),
            ("src/exl3_forward/psk.rs", 8),
            ("src/exl3_forward/reprime.rs", 3),
            ("src/exl3_forward/rounds.rs", 1),
            ("src/exl3_forward/router_r16.rs", 6),
            ("src/exl3_forward/scratch.rs", 2),
            ("src/exl3_forward/spec_pass.rs", 3),
            ("src/exl3_forward/tensors.rs", 6),
            ("src/exl3_forward/tune_hc.rs", 2),
            ("src/exl3_forward/verify.rs", 2),
            ("src/exl3_forward/widths.rs", 5),
            ("src/exl3_forward/wp21.rs", 9),
            ("src/exl3_forward/wp_levers.rs", 15),
            ("src/exl3_forward/flash256_stage_tests.rs", 2), // SPLIT1 exl3_forward: a moved test module (no cfg(test) wrapper in its own file) is counted raw
            ("src/exl3_forward/dres.rs", 10), // LANEWIDE-MID (2026-11-13): 6 -> 10, the --probe-mid-gemm report lines. LANEWIDE-DRES (2026-11-12): the --probe-dense-resident report lines (dprintln!, probe-only)
            ("src/exl3_forward/lwv.rs", 17), // LANEWIDE-EXACT (2026-11-13): 9 -> 17, the XCLASS_TAPS / XCLASS_HID / XCLASS_TIME / XCLASS_SEGSTART and --probe-lane-wide-pin report lines (probe-only). LANEWIDE (2026-11-13): 6 -> 9, the --probe-lane-wide-xclass report lines (probe-only). LANEWIDE (2026-11-12): the --probe-lane-wide-cost report lines (dprintln!, probe-only; output IS the product)
            ("src/exl3_forward/cpv.rs", 7), // COPYPF1 (2026-11-12): the --probe-copy-prefill report lines (dprintln!, probe-only; output IS the product)
            ("src/exl3_forward/dense.rs", 12),
            ("src/exl3_forward/dhead.rs", 11),
            ("src/exl3_forward/w4s.rs", 8),
            ("src/exl3_forward/wp16.rs", 12),
            ("src/exl3_forward/xtp/xtp_ref.rs", 42), // S-B9-REL-INTEG A5: +1, the harness's RANK_IDENTITY FAIL line when pass 1 errors (a probe, not a served path). SPLIT1: the former xtp.rs row (75) now spans these five files
            ("src/exl3_forward/xtp/tp_state.rs", 17),
            ("src/exl3_forward/xtp/slice_probe.rs", 8),
            ("src/exl3_forward/xtp/tp_config.rs", 5),
            ("src/exl3_forward/xtp/spine.rs", 3),
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
            ("src/exl3_ppl/mod.rs", 4),
            ("src/exl3_regime.rs", 1), // S-B9-REL-QSAREGIME: the probe prints through one closure (not a served path)
            // MTPREPROBE2 W1.5: +2 = the two step_plain a-xcheck record-failure branches
        // (eprintln, matching the existing sibling site in finish()/xcheck drain).
        // S-B9-REL-SMALLFIXES item 1: 87 -> 86, the two MTP boot-banner println! became one
        // (the line is built by the testable mtp_banner()).
        ("src/exl3_serve/mod.rs", 3), // COPYPF1 (2026-11-12): 107 -> 108, the [round-trace] copy-prefill line behind --round-trace (diagnostic; the tree already read 107 at cb891dbe, drift not re-pinned then: re-pinned consciously). MTPAGREE1 (v0.7.7): 103 -> 106, the [df2-tp2] lane dump on the lockstep-failure arm, the [round-trace] line behind --round-trace and the [tp-ident] differing-digest line (failure / diagnostic paths; TOPKM1 hygiene, conscious update). Earlier: EXL27C: 88 -> 97 (--df2-trace gated dispatch/round prints; conscious update). +1 AGREE16: the width drill's line (test drill; one-shot since AGREE16B, right before a link abort)
        // SPLIT1: src/exl3_serve.rs (108) became src/exl3_serve/*.rs. The rows below sum to 113, not 108: the old counter's brace match of one inline test module
        // lost its string parity and swallowed the code to the end of the file (1 print, prime_batch_boot_log), and the 4 prints of three former inline test
        // modules (the dprintln!/eprintln! lines of mtp_switch_tests / stop_info_tests / token_logprobs_tests) are now whole files, which the cfg(test)-mod stripper cannot skip.
        ("src/exl3_serve/admission.rs", 2),
        ("src/exl3_serve/builder.rs", 32),
        ("src/exl3_serve/copy.rs", 2),
        ("src/exl3_serve/df2.rs", 2),
        ("src/exl3_serve/lockstep.rs", 3),
        ("src/exl3_serve/mtp_round.rs", 5),
        ("src/exl3_serve/mtp_switch_tests.rs", 1),
        ("src/exl3_serve/prefill.rs", 3),
        ("src/exl3_serve/scheduler.rs", 7),
        ("src/exl3_serve/serve.rs", 13),
        ("src/exl3_serve/spec_program.rs", 15),
        ("src/exl3_serve/stop_info_tests.rs", 2),
        ("src/exl3_serve/token_logprobs_tests.rs", 1),
        ("src/exl3_serve/tp_serve.rs", 15),
        ("src/exl3_serve/wp24_gate.rs", 7),
            ("src/exl3_tune/boot.rs", 7), // SPLIT1 exl3_tune: the old row "src/exl3_tune.rs" (7 outside cfg(test)) moved with the boot code; tests.rs prints nothing
            ("src/exl3_wp27.rs", 8),
            // SPLIT1: src/gptq.rs split into src/gptq/*.rs; the 42 prints are spread over the rows below.
            ("src/gptq/calib.rs", 8),
            ("src/gptq/dflash2.rs", 7),
            ("src/gptq/igs.rs", 5),
            ("src/gptq/io.rs", 2),
            ("src/gptq/lmhead.rs", 4),
            ("src/gptq/mod.rs", 4),
            ("src/gptq/run.rs", 12),
            ("src/gpu/attn_batch.rs", 5), // SPLIT1 gpu: the old row "src/gpu.rs" (323 outside cfg(test)) is split per file below; the moved test file shard_raw_real_tests.rs (no cfg(test) wrapper in its own file) is counted raw: +3, so the sum is 326
            ("src/gpu/bandwidth.rs", 11),
            ("src/gpu/bench_mtp.rs", 8),
            ("src/gpu/bench_tree.rs", 4),
            ("src/gpu/construct.rs", 3),
            ("src/gpu/debug.rs", 7),
            ("src/gpu/decode.rs", 2),
            ("src/gpu/df2_capture.rs", 2),
            ("src/gpu/e9.rs", 3),
            ("src/gpu/forward_batch.rs", 2),
            ("src/gpu/gates.rs", 1),
            ("src/gpu/gemm_policy.rs", 1),
            ("src/gpu/gemm_prefill.rs", 15),
            ("src/gpu/generate.rs", 20),
            ("src/gpu/gptq_gpu.rs", 1),
            ("src/gpu/gptq_rot.rs", 1),
            ("src/gpu/graphs.rs", 2),
            ("src/gpu/launch_gemm.rs", 2),
            ("src/gpu/load.rs", 22),
            ("src/gpu/moe_batch.rs", 2),
            ("src/gpu/mtp.rs", 5),
            ("src/gpu/pool.rs", 28),
            ("src/gpu/prefill.rs", 17),
            ("src/gpu/probes_gemm.rs", 31),
            ("src/gpu/probes_mxfp4.rs", 46),
            ("src/gpu/probes_verify.rs", 27),
            ("src/gpu/qsa.rs", 4),
            ("src/gpu/qwen4_exp.rs", 2),
            ("src/gpu/shard_ops.rs", 1),
            ("src/gpu/shard_raw_real_tests.rs", 3),
            ("src/gpu/tp_comm.rs", 5),
            ("src/gpu/tp_shard.rs", 22),
            ("src/gpu/verify.rs", 4),
            ("src/gpu/w4a4.rs", 5),
            ("src/gpu/xchain.rs", 12),
            ("src/inkling2/actq/probe.rs", 16), // INKLING2 Q1: --probe-inkling2-actq report lines (probe class)
            ("src/inkling2/chat/tests.rs", 2), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/cpu.rs", 9), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/gpu/k1_probe.rs", 10), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/gpu/probe.rs", 12), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/model/mod.rs", 1), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/model/probe.rs", 17), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/model/probe_s7.rs", 12), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/model/serve.rs", 1), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/mtp/gpu.rs", 1), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/mtp/probe.rs", 17), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/pack.rs", 12), // INKLING2: --probe-inkling2-pack census lines (probe class)
            ("src/inkling2/serve/http.rs", 1), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/serve/tests.rs", 1), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/tok.rs", 10), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/tp/launch.rs", 3), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/tp/sim.rs", 10), // INKLING2 (conscious update): probe / boot / fatal-exit lines of the new family
            ("src/inkling2/vision/probe.rs", 16), // INKLING2 S5: --probe-inkling2-vision report lines (probe class)
            ("src/json_schema/mod.rs", 1),
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
            ("src/net/mod.rs", 14), // +1 AGREE16: the width-refusal line of agree_ext (printed once, right before the link abort)
            // S-B9-REL-FINALFIX F7 (2026-10-11): opts.rs = 2 direct prints.
            // 1 = the FATAL env-var rejection line at boot (served, pre-router, by design);
            // +1 = the #[ignore]d f7_differential long-run progress line (test-only).
            // SPLIT1 opts: the old row (3) is split per file: mod.rs = the FATAL env-var rejection line (1), tests.rs = the two
            // test-code lines (the #[ignore]d f7_differential progress line and the f9_differential count line, 2; counted raw
            // now that the test module is its own file). Sum unchanged: 3.
            ("src/opts/mod.rs", 1),
            ("src/opts/tests.rs", 2), // F9: +1 the f9_differential count line, test code read with --nocapture
            ("src/pp.rs", 14),
            // SPLIT1: src/quant.rs split into src/quant/*.rs; the 5 prints are spread over the rows below (sum now 7: a former inline test module became a whole file and its prints count).
            ("src/quant/q2_tests.rs", 2),
            ("src/quant/recipe.rs", 5),
            ("src/qwen.rs", 7),
            // S-B9-REL-BENCHFIX2: +2 = the "[serve] sse keep-alive: ..." boot line (println!, the
            // one-time banner class, printed by http_listen before the socket exists) and the
            // FATAL line of fatal_option (direct by the H7 rule above: printed, then flush_and_exit)
            ("src/server/mod.rs", 3),
            ("src/server/chat_params.rs", 2),
            ("src/server/engine.rs", 1),
            ("src/server/helpers.rs", 13),
            ("src/server/reasoning.rs", 1),
            ("src/server/sse.rs", 1),
            ("src/shard_plan.rs", 9), // FP4L (2026-10-29): +1, the `fp4l shard opens` line of the --probe-shard-plan FP4L arm (probe path). NAIVE: +1, the `owner mismatch` line of the Naive arm of --probe-shard-plan (probe path)
            ("src/exl3_forward/fp4l_fwd/probe_gpu.rs", 13), // FP4L: the --probe-fp4l-gpu verdict lines (probe path; the served FP4L paths log via rprintln!/drprintln!); +1 (2026-10-30) FP4L_GPU_PAIR // SPLIT1 fp4l_fwd: all 13 prints sit in probe_gpu.rs
            ("src/fp4l/probe.rs", 29), // FP4L-0 (2026-10-29): the --probe-fp4l CPU receipts print to stdout (probe verdict lines; never a served path); +2 (2026-10-30): the FP4L_PROBE_FAIL / FP4L_GPU_FAIL exit lines of the probe dispatch (moved out of main.rs).
            // BASICS (owner 2026-11-02): the version line (stdout, or stderr under --print-config) and the
            // configuration block: boot banners, which stay direct by this module's rule.
            ("src/startup.rs", 3),
            // SPLIT1 tokenizer: the old row "src/tokenizer.rs" (9, the inline test module stripped by the counter) is split per file below; the moved test file tests.rs has no cfg(test) wrapper in its own file and is counted raw: +14 (the wp01 / template test prints). 2+4+3 = 9 outside tests.
            ("src/tokenizer/jinja.rs", 2),
            ("src/tokenizer/qwen.rs", 4),
            ("src/tokenizer/stream.rs", 3),
            ("src/tokenizer/tests.rs", 13),
            // S-B9-REL-SURFACE2 item 10: + 2, the ignored `tool_quote_probe` instrument (test code).
            // S-B9-REL-SURFACE2B R1: + 5, the count tables of `exhaustive_small_blocks` and the
            // timing line of the 1 MB test (test code, read with --nocapture).
            // S-B9-TOOLPARSE3: + 4 in tools.rs (the megabyte-block and shadow-cost instrument lines, test
            // code, read with --nocapture), and src/tools/cost.rs: 32, the measurement tables of the
            // cost model (ignored instruments and one pinned test, test code only).
            // SPLIT1 tools: the old row "src/tools.rs" (15) is split per file: 2 in calls.rs, 13 in the moved test module tests.rs (counted raw; the old brace-match did not strip `pub(in crate::tools) mod tests`, so the old 15 held them too).
            ("src/tools/calls.rs", 2),
            ("src/tools/tests.rs", 13),
            // SPLIT1 tools/cost: the old row "src/tools/cost.rs" (32) is all in the moved test module; mod.rs has none.
            ("src/tools/cost/tests.rs", 32),
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
            ("src/main.rs", 156), // INKLING2 (2026-11-22): 143 -> 156, the --probe-inkling2-* dispatches' FAIL-path / result lines (probe-only; conscious update). DSV41 INT1: 142 -> 143, the FAIL-path eprintln of the --dsv41-engram-pack/-verify/-iobench dispatch (ENGb; CLI tool, no served path; conscious update). NIGHT1 (2026-11-17): 1644 -> 1645, the FAIL-path eprintln of the --probe-gpu-wait dispatch (probe-only). V0710 port (2026-11-14): 1644 counted by this test on the merged tree (master 1637 + the fork's probe-only FAIL-path eprintlns). SELFUPD: 1632 -> 1637 (--update/--no-update-check help + notices). Fork history: ULTRA merge of lanewide-tp (2026-11-13): 1623 -> 1624, LANEWIDE-EXACT the FAIL-path eprintln of --probe-lane-wide-pin (probe-only). ULTRA merge (2026-11-13): PACKPF +1 and LANEWIDE-MID / XCLASS +2 (the FAIL-path eprintlns of --probe-mid-gemm and --probe-lane-wide-xclass, probe-only) = 1623. PACKPF (2026-11-13): 1620 -> 1621, the FAIL-path eprintln of --probe-exl3-packpf (probe-only). LANEWIDE-DRES (2026-11-12): 1619 -> 1620, the FAIL-path eprintln of --probe-dense-resident (probe-only). LANEWIDE (2026-11-12): 1618 -> 1619, the FAIL-path eprintln of the --probe-lane-wide-cost dispatch (probe-only). COPYPF1 (2026-11-12): 1617 -> 1618, the FAIL-path eprintln of the --probe-copy-prefill dispatch (probe-only; the tree already read 1617 at cb891dbe, drift not re-pinned then: re-pinned consciously). BASICS (2026-11-02): 1607 -> 1616, the --version print, the help's version line (+1: the old first line split in two), the --version help entry and six --quantize recipe-example help lines (one-time banner/help class; conscious update). PR#27-D3 (2026-10-30): +4 - 3 help lines + the FAIL-path eprintln in the probe dispatch (probe-only) // TOPKM1 hygiene (2026-10-28): 1595 -> 1603, the --quantize recipe help text (b612145e: +9/-1 println! in the usage printer; conscious update). Earlier: OPERATIVE at the port merge: 1547 + 33 (naive) + 15 (master) = 1595 on the merged tree. (A second, stale `src/main.rs` row of 1580 sat below this one; `find` takes the first row, so it never applied — removed.)
            // SPLIT1 main: the old row "src/main.rs" (1645 outside cfg(test)) is split per file below (main.rs keeps `fn main`, the 142 above; the rest moved to src/cli/*, sum unchanged). The row comments here and below are the history of the whole former file.
            ("src/cli/args.rs", 21),
            ("src/cli/bench_basic.rs", 78),
            ("src/cli/bench_df2.rs", 18),
            ("src/cli/bench_mtp.rs", 56),
            ("src/cli/dsv4_serve.rs", 43),
            ("src/cli/dsv4_serve/dspark_driver.rs", 43),
            ("src/cli/help.rs", 597),
            ("src/cli/probe_df2_draft.rs", 53),
            ("src/cli/probe_df2_misc.rs", 33),
            ("src/cli/probe_df2_round.rs", 66),
            ("src/cli/probe_dspark.rs", 83),
            ("src/cli/probe_dspark_round.rs", 27),
            ("src/cli/probe_tq.rs", 18),
            ("src/cli/probes_dsv4.rs", 60),
            ("src/cli/probes_serve.rs", 10),
            ("src/cli/probes_state.rs", 34),
            ("src/cli/quantize.rs", 36),
            ("src/cli/replay.rs", 19),
            ("src/cli/serve.rs", 54),
            ("src/cli/stq.rs", 19),
            ("src/cli/tp_cluster.rs", 52), // INKLING2 TP2: +1 the node-side `inkling2 node` fatal line (exit path, like dsv41 above); // DSV41 TPC: the node-side `node dsv41 tp error` line (eprintln! then exit(1)) in the dsv41 node branch: a fatal path that stays direct by the logq policy
            ("src/cli/tp_heads.rs", 55),
            ("src/cli/tp_serve_trunk.rs", 30),
        ("src/self_update.rs", 23), // SELFUPD: updater is an offline-exit path (no serving), direct prints intended
            // NAIVE history of src/main.rs: +33, the --probe-naive-* dispatches' NAIVE_*_FAIL exit-path lines, their usage expect()s' siblings, and the head/node Naive TP boot lines (probe/boot classes)
            // Conscious updates, not conversion misses.
        ("src/tp.rs", 13),
            ("src/tp_bench/mod.rs", 37), // SPLIT1: the former tp_bench.rs row (75) now spans these three files
            ("src/tp_bench/nway.rs", 28),
            ("src/tp_bench/reduce_w4.rs", 10),
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
            // SPLIT1: src/naive/model.rs split into src/naive/model/*.rs; the 33 prints are spread over the rows below (sum now 34: a former inline test module became a whole file and its prints count).
            // (old row comment: // LOADFAST (f2571038): the per-layer routed up_proj scale println! removed, the layer-loaded line is a dprintln! (34 -> 33; TOPKM1 hygiene, conscious update))
            ("src/naive/model/load.rs", 3),
            ("src/naive/model/probes.rs", 30),
            ("src/naive/model/t0c_tests.rs", 1),
            ("src/naive/pack.rs", 11),
            // SPLIT1: src/naive/serve.rs split into src/naive/serve/*.rs; the 68 prints are spread over the rows below.
            ("src/naive/serve/engine.rs", 6),
            ("src/naive/serve/flags.rs", 2),
            ("src/naive/serve/head.rs", 11),
            ("src/naive/serve/loops.rs", 5),
            ("src/naive/serve/probes.rs", 44),
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
