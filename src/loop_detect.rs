//! WP08 — streaming loop detector, a line-for-line port of the rival's
//! `exllamav3/generator/loop_detect.py` (vcruz305/exllamav3 @523ecd3), including its heap wake
//! schedule, so a stream is cut at exactly the token where the rival cuts it.
//!
//! Wiring (the rival's job.py:312-318 and 1010-1013): `LoopDetector::new(window, window / min_reps)`
//! per request, fed every emitted token that did not already end the request (stop token /
//! max_new); a `true` from `feed` ends the response as a normal stop with reason `loop_detected`.
//! The rival's launcher default is `stop_on_loop = (300, 3)` (chat.py -lw 300 -lmr 3).
//!
//! Condition: the WHOLE last-`W` window is periodic with some period p <= max_period
//! (`s[i] == s[i-p]` for the newest `W - p` positions). Each period keeps a consecutive-match
//! streak; a detector whose streak breaks sleeps `W` tokens (the mismatch has to leave the window
//! before it can fire again) and re-derives its streak with a backlog scan on waking. Cost per fed
//! token: the heap pops of the detectors due at that token (amortized O(active · log P)); no
//! allocation after construction.
//!
//! Cross-check: `tests/fixtures/loop_detect_ref.py` (the rival's code, torch-free; verified
//! identical to the rival file on all its cases) generates `loop_detect_vectors.txt`, which the
//! unit test below replays bit-for-bit (first detection index, count, index sum, period).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Streaming loop detector (see the module docs). Tokens are opaque ids.
pub struct LoopDetector {
    w: usize,
    max_period: usize,
    buf: Vec<u32>,
    total: usize,
    streak: Vec<usize>,
    // (wake_time, period, generation), min-first: Python heapq's tuple order.
    heap: BinaryHeap<Reverse<(usize, usize, u64)>>,
    gen: Vec<u64>,
    detected_period: Option<usize>,
}

impl LoopDetector {
    /// The rival's `LoopDetector(window_size, max_period)`; `max_period = None` means W / 3.
    /// The period is capped at W / 2 (no longer period can repeat inside the window).
    pub fn new(window: usize, max_period: Option<usize>) -> Self {
        let w = window.max(2);
        let max_period = max_period.filter(|&p| p > 0).unwrap_or(w / 3).min(w / 2);
        let mut heap = BinaryHeap::with_capacity(4 * max_period + 8);
        for p in 1..=max_period {
            heap.push(Reverse((w, p, 0u64)));
        }
        Self {
            w,
            max_period,
            buf: vec![0; w],
            total: 0,
            streak: vec![0; max_period + 1],
            heap,
            gen: vec![0; max_period + 1],
            detected_period: None,
        }
    }

    /// The job wiring: `LoopDetector(window_size, window_size // min_reps)` (job.py:318).
    pub fn for_stop_on_loop(window: usize, min_reps: usize) -> Self {
        Self::new(window, Some(window / min_reps.max(1)))
    }

    /// The rival's `period` property: the period of the last detector that fired (not
    /// necessarily the fundamental one; see `fundamental_period`).
    pub fn period(&self) -> Option<usize> {
        self.detected_period
    }

    /// Tokens fed so far.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Diagnostics: the smallest p <= max_period for which the current window is p-periodic.
    /// O(W · P) — call once, at detection, for the log line.
    pub fn fundamental_period(&self) -> Option<usize> {
        if self.total < self.w {
            return None;
        }
        let t = self.total;
        let at = |k: usize| self.buf[k % self.w];
        (1..=self.max_period).find(|&p| (0..self.w - p).all(|k| at(t - 1 - k) == at(t - 1 - k - p)))
    }

    fn schedule(&mut self, p: usize, wake: usize) {
        self.gen[p] += 1;
        self.heap.push(Reverse((wake, p, self.gen[p])));
    }

    fn backlog_scan(&self, p: usize) -> usize {
        let t = self.total;
        let w = self.w;
        let mut streak = 0;
        for k in 0..(w - p) {
            if self.buf[(t - 1 - k) % w] == self.buf[(t - 1 - k - p) % w] {
                streak += 1;
            } else {
                break;
            }
        }
        streak
    }

    /// Feed one token; true while a loop is detected (the rival's `feed`).
    pub fn feed(&mut self, token: u32) -> bool {
        let w = self.w;
        self.buf[self.total % w] = token;
        self.total += 1;
        if self.total < w {
            return false;
        }
        let t = self.total;
        let mut detected = false;
        while let Some(&Reverse((wake, p, g))) = self.heap.peek() {
            if wake > t {
                break;
            }
            self.heap.pop();
            if g != self.gen[p] {
                continue; // stale entry
            }
            if self.streak[p] > 0 {
                // active detector: check the newest token
                if self.buf[(t - 1) % w] == self.buf[(t - 1 - p) % w] {
                    self.streak[p] += 1;
                    if self.streak[p] >= w - p {
                        self.detected_period = Some(p);
                        detected = true;
                    }
                    self.schedule(p, t + 1);
                } else {
                    // streak broken: sleep until the mismatch leaves the window
                    self.streak[p] = 0;
                    if self.detected_period == Some(p) {
                        self.detected_period = None;
                    }
                    self.schedule(p, t + w);
                }
            } else {
                // waking from sleep: re-derive the streak from the buffer
                let streak = self.backlog_scan(p);
                self.streak[p] = streak;
                if streak >= w - p {
                    self.detected_period = Some(p);
                    detected = true;
                    self.schedule(p, t + 1);
                } else if streak > 0 {
                    self.schedule(p, t + 1);
                } else {
                    self.schedule(p, t + w);
                }
            }
        }
        detected || self.detected_period.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::LoopDetector;

    /// splitmix64 — the generator of tests/fixtures/loop_detect_ref.py.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u32 {
            (self.next() % n) as u32
        }
    }

    fn build(spec: &str) -> (usize, usize, Vec<u32>) {
        let mut parts = spec.split(';');
        let head: Vec<u64> = parts.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
        let (w, reps) = (head[0] as usize, head[1] as usize);
        let mut rng = Rng(head[2]);
        let mut toks = Vec::new();
        for seg in parts {
            let f: Vec<&str> = seg.split(':').collect();
            let n: usize = f[1].parse().unwrap();
            match f[0] {
                "R" => {
                    let v: u64 = f[2].parse().unwrap();
                    for _ in 0..n { toks.push(rng.below(v)); }
                }
                "P" | "Q" => {
                    let p: usize = f[2].parse().unwrap();
                    let v: u64 = f[3].parse().unwrap();
                    let pat: Vec<u32> = (0..p).map(|_| rng.below(v)).collect();
                    let k: usize = if f[0] == "Q" { f[4].parse().unwrap() } else { 0 };
                    for i in 0..n {
                        if k > 0 && i % k == k - 1 { toks.push(rng.below(v)); } else { toks.push(pat[i % p]); }
                    }
                }
                other => panic!("bad segment {other}"),
            }
        }
        (w, reps, toks)
    }

    fn summarize(w: usize, reps: usize, toks: &[u32]) -> (i64, usize, usize, i64) {
        let mut d = LoopDetector::for_stop_on_loop(w, reps);
        let (mut first, mut count, mut sum, mut period) = (-1i64, 0usize, 0usize, -1i64);
        for (i, &t) in toks.iter().enumerate() {
            if d.feed(t) {
                if first < 0 {
                    first = i as i64;
                    period = d.period().map(|p| p as i64).unwrap_or(-1);
                }
                count += 1;
                sum += i;
            }
        }
        (first, count, sum, period)
    }

    /// Bit-for-bit replay of the rival reference (tests/fixtures/loop_detect_ref.py).
    #[test]
    fn loop_detect_matches_rival_reference() {
        let vectors = include_str!("../tests/fixtures/loop_detect_vectors.txt");
        let mut n = 0;
        for line in vectors.lines().filter(|l| !l.trim().is_empty()) {
            let (spec, want) = line.split_once(" | ").unwrap();
            let (w, reps, toks) = build(spec);
            let got = summarize(w, reps, &toks);
            let got_s = format!("first={} count={} sum={} period={}", got.0, got.1, got.2, got.3);
            assert_eq!(got_s, want.trim(), "vector {spec}");
            n += 1;
        }
        assert!(n >= 20, "only {n} vectors parsed — the fixture is missing or truncated");
    }

    /// WP08 gate: periods 1, 2, 7, 100 detect at (300, 3); 101 never does.
    #[test]
    fn loop_detect_periods() {
        for (p, fires) in [(1usize, true), (2, true), (7, true), (100, true), (101, false)] {
            let pat: Vec<u32> = (0..p as u32).map(|i| 1000 + i * 7).collect();
            let mut d = LoopDetector::for_stop_on_loop(300, 3);
            let mut hit = None;
            for i in 0..1500 {
                if d.feed(pat[i % p]) { hit = Some(i); break; }
            }
            assert_eq!(hit.is_some(), fires, "period {p}");
            if let Some(i) = hit {
                // a clean periodic stream fires once the whole window repeats
                assert_eq!(i, 299, "period {p} fired at {i}");
                assert_eq!(d.fundamental_period(), Some(p));
            }
        }
    }

    /// One mismatch resets: the loop must be re-established over a full window after it.
    #[test]
    fn loop_detect_one_mismatch_resets() {
        let mut d = LoopDetector::for_stop_on_loop(300, 3);
        let pat = [5u32, 6, 7, 8, 9];
        let mut toks: Vec<u32> = (0..250).map(|i| pat[i % 5]).collect();
        toks.push(424242);
        let resume = toks.len();
        toks.extend((0..600).map(|i| pat[i % 5]));
        let first = toks.iter().position(|&t| d.feed(t)).unwrap();
        assert!(first >= resume + 299 - 5, "fired at {first}, before a clean window after the mismatch");
        assert!(first <= resume + 299, "fired at {first}, later than the whole window");
    }

    /// 10K random tokens never fire.
    #[test]
    fn loop_detect_random_never_fires() {
        let mut rng = Rng(0xC0FFEE);
        let mut d = LoopDetector::for_stop_on_loop(300, 3);
        for _ in 0..10_000 {
            assert!(!d.feed(rng.below(248_320)));
        }
    }
}
