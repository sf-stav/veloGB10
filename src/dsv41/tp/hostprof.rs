//! DSV4.1 TP (leg TX1 item 3): an opt-in, in-process host-time profile of a decode / verify round
//! (`--dsv41-host-profile on`), the untraced measurement of the time the host spends between graphs (traces inflate host
//! gaps: AGENTS section 5). `mark(i)` stamps `Instant`s into a thread-local row; `round_end` folds the row into the
//! per-segment samples and prints p50 / mean / p99 of every segment each `EVERY` rounds. Off (the default) a mark is one
//! relaxed atomic load. Nothing here touches the GPU or any bit of the output.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// marks, in the order a round passes them (segment k = mark k+1 - mark k; segment 0 starts at the previous round's end)
pub const M_ENTRY: usize = 0;       // round() entered
pub const M_DRAFTED: usize = 1;     // draft + ids download + the count decision (t1)
pub const M_PLANNED: usize = 2;     // forward_segs: plan_call + upload_rows + engram rows done
pub const M_GRAPH: usize = 3;       // forward_segs: graph_call (tables + replay enqueue) returned
pub const M_FWD_DONE: usize = 4;    // forward_segs returned (engram check, lane bookkeeping, head read)
pub const M_ENQUEUED: usize = 5;    // head logits / cut / verify_k / main_rows / commit enqueued; about to read the lanes
pub const M_SYNCED: usize = 6;      // get_lanes + out_tokens returned (the round's GPU work is complete)
pub const M_END: usize = 7;         // the round's host bookkeeping done
const N: usize = 8;
const NAMES: [&str; N] = ["since the last round ended (serve loop, sampler, agree)", "set_lanes + draft + count decision", "plan + upload rows + engram rows",
                          "graph tables + replay enqueue", "after replay: bookkeeping + head read", "head / cut / verify_k / main_rows / commit enqueue",
                          "wait for the round's GPU work (get_lanes)", "post: rollback, stats, receipts"];
const EVERY: usize = 256;

static ON: AtomicU8 = AtomicU8::new(2);   // 2 = unread
pub fn on() -> bool {
    match ON.load(Ordering::Relaxed) {
        2 => {
            let v = crate::opts::get(crate::opt!("dsv41-host-profile")).map_or(false, |s| s.trim() == "on");
            ON.store(v as u8, Ordering::Relaxed);
            v
        }
        v => v == 1,
    }
}

struct Acc { cur: [Option<Instant>; N], last_end: Option<Instant>, rows: Vec<[f64; N]>, rounds: u64 }
thread_local! { static ACC: RefCell<Acc> = RefCell::new(Acc { cur: [None; N], last_end: None, rows: Vec::new(), rounds: 0 }); }

pub fn mark(i: usize) {
    if !on() { return; }
    let now = Instant::now();
    ACC.with(|a| a.borrow_mut().cur[i] = Some(now));
}

/// Fold the finished round's marks into the samples (`rank` for the printed line). A round with a missing mark is dropped.
pub fn round_end(rank: usize, _rows: usize) {
    if !on() { return; }
    mark(M_END);
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let cur = a.cur;
        let prev_end = a.last_end;
        a.last_end = cur[M_END];
        a.cur = [None; N];
        let (Some(prev), true) = (prev_end, cur.iter().all(|c| c.is_some())) else { return };
        let t = |i: usize| cur[i].unwrap();
        let ms = |x: Instant, y: Instant| y.saturating_duration_since(x).as_secs_f64() * 1e3;
        let mut r = [0f64; N];
        r[0] = ms(prev, t(M_ENTRY));
        for k in 1..N { r[k] = ms(t(k - 1), t(k)); }
        a.rows.push(r);
        a.rounds += 1;
        if a.rows.len() >= EVERY {
            let n = a.rows.len();
            let mut out = format!("[dsv41 hostprof] rank {rank}: {n} rounds ({} so far), host ms per segment (p50 / mean / p99); the sum of the non-wait segments is the host time between GPU work:", a.rounds);
            let mut busy = 0.0;
            for k in 0..N {
                let mut v: Vec<f64> = a.rows.iter().map(|r| r[k]).collect();
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let mean = v.iter().sum::<f64>() / n as f64;
                out.push_str(&format!("\n    {:>2} {:<55} {:7.3} / {:7.3} / {:7.3}", k, NAMES[k], v[n / 2], mean, v[(n * 99 / 100).min(n - 1)]));
                if k != M_SYNCED { busy += mean; }
            }
            let tot: f64 = (0..N).map(|k| a.rows.iter().map(|r| r[k]).sum::<f64>() / n as f64).sum();
            out.push_str(&format!("\n    host segments except the GPU wait: mean {busy:.3} ms of a {tot:.3} ms round period"));
            crate::reprintln!("{out}");
            a.rows.clear();
        }
    });
}
