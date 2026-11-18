//! TURNRESUME: `--turn-resume on|off` and `--next-turn-prefill on|off|inline` (both default off).
//!
//! What the engine did before (measured 2026-11-12, `.15`, Flash-Next 3.05bpw, thinking on, agent turns with a tool
//! catalogue): every follow-up already resumed at the previous prompt's end (the per-lane prompt-end snapshot,
//! `cached = previous prompt - 1`), because the next turn re-renders the previous prompt verbatim up to its generation
//! prompt. What it re-read every turn was the previous REPLY as the client sends it back (the answer without its
//! reasoning, its tool calls, the closing tokens) plus the new turn.
//!
//! This module is the engine-independent half:
//! - the server, once a reply has gone out, renders the history the way the NEXT request will send it (the reply
//!   without reasoning, or with it when the client echoes reasoning back) and posts those token ids here (`post`);
//! - the scheduler drains them (`take`) and, while every lane is idle, prefills the rendered reply onto the lane's
//!   pre-turn state in small chunks (`--next-turn-chunk`), yielding to any arriving request between chunks; the
//!   result is kept as a second per-lane state next to the prompt-end snapshot (that one is never touched, so a wrong
//!   guess costs nothing but idle time);
//! - at admission the deepest state that the new prompt starts with wins (`pick`).
//!
//! OUTPUT-CHANGING (the class of the C1 tail checkpoints): a resumed request runs a different chunk grid than one that
//! reads the reply as part of its own prefill, and this model amplifies rounding chaotically, so bytes differ from
//! flags-off (and from a cold read, as every prefix-cache resume already does). A request whose prompt does not start
//! with the kept state resumes exactly as with the flags off.
//!
//! Counters (`velogb10_turn_*` on /metrics) live here; the arms live in exl3_serve.rs (EXL3) and post from server.rs.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// `--next-turn-prefill` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NtMode {
    Off,
    /// background passes at idle time
    On,
    /// diagnostic: the same pass, run synchronously at the next admission (gate: identical bytes to `On`)
    Inline,
}

impl NtMode {
    pub fn parse(s: &str) -> NtMode {
        match s { "on" | "1" | "true" => NtMode::On, "inline" => NtMode::Inline, _ => NtMode::Off }
    }
    pub fn name(self) -> &'static str {
        match self { NtMode::Off => "off", NtMode::On => "on", NtMode::Inline => "inline" }
    }
}

/// The server posts hints only while a scheduler that consumes them is running (set at boot).
static ACCEPTING: AtomicBool = AtomicBool::new(false);
static HINTS: Mutex<Vec<(Vec<u32>, std::time::Instant)>> = Mutex::new(Vec::new());
/// At most this many hints wait (the oldest go first): a hint is only useful for the conversation that just ended.
const MAX_HINTS: usize = 16;

pub fn set_accepting(on: bool) { ACCEPTING.store(on, Relaxed); }
pub fn accepting() -> bool { ACCEPTING.load(Relaxed) }

/// A rendered next-turn prefix (token ids) for the conversation whose reply just finished.
pub fn post(toks: Vec<u32>) {
    if !accepting() || toks.is_empty() { return; }
    let mut q = HINTS.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() >= MAX_HINTS { q.remove(0); HINTS_DROPPED.fetch_add(1, Relaxed); }
    q.push((toks, std::time::Instant::now()));
    HINTS_POSTED.fetch_add(1, Relaxed);
}

pub fn pending() -> bool { !HINTS.lock().unwrap_or_else(|e| e.into_inner()).is_empty() }

/// The posted hints with the time each was posted (a background pass starts --next-turn-delay-ms after it).
pub fn take() -> Vec<(Vec<u32>, std::time::Instant)> { std::mem::take(&mut *HINTS.lock().unwrap_or_else(|e| e.into_inner())) }

/// Which free lane a hint belongs to: the lane whose prompt-end key is the longest prefix of the hint (ties: the
/// lowest slot). `keys[i]` = None for a busy lane or a lane without a snapshot.
pub fn lane_for(hint: &[u32], keys: &[Option<&[u32]>]) -> Option<usize> {
    keys.iter().enumerate()
        .filter_map(|(i, k)| k.filter(|k| !k.is_empty() && k.len() <= hint.len() && hint[..k.len()] == k[..]).map(|k| (k.len(), std::cmp::Reverse(i))))
        .max().map(|(_, std::cmp::Reverse(i))| i)
}

/// Admission: how a kept next-turn state (`kept`: the tokens its rows hold) relates to the new prompt key
/// (`key`, `end` = the rows a prefill may cover, plen - 1) and the prompt-end resume depth `snap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// resume from the kept state at this many tokens
    Use(usize),
    /// no kept state on this lane
    None,
    /// the prompt leaves the kept state before its end (the client sent something else: a wrong guess, an edit, a branch)
    Diverged,
    /// the kept state is not deeper than the prompt-end snapshot / reaches past the prompt's prefill end
    NotDeeper,
}

pub fn pick(kept: Option<&[u32]>, key: &[u32], end: usize, snap: usize) -> Pick {
    let Some(k) = kept else { return Pick::None };
    if k.is_empty() { return Pick::None; }
    let lcp = k.iter().zip(key).take_while(|(a, b)| a == b).count();
    if lcp < k.len() { return Pick::Diverged; }
    if k.len() > end || k.len() <= snap { return Pick::NotDeeper; }
    Pick::Use(k.len())
}

/// The chunk ends of a background pass over [from, to) at `chunk` rows.
pub fn bg_grid(from: usize, to: usize, chunk: usize) -> Vec<usize> {
    let c = chunk.max(1);
    let mut v = Vec::new();
    let mut p = from;
    while p < to { p = (p + c).min(to); v.push(p); }
    v
}

// ---------------------------------------------------------------- counters (/metrics velogb10_turn_*)
pub static HINTS_POSTED: AtomicU64 = AtomicU64::new(0);
pub static HINTS_DROPPED: AtomicU64 = AtomicU64::new(0);
/// hints that found no lane: [no free lane holds the conversation, image prompt, rendered prefix does not start
/// with the lane's prompt]
pub static HINT_NOLANE: AtomicU64 = AtomicU64::new(0);
pub static HINT_MISMATCH: AtomicU64 = AtomicU64::new(0);
pub static BG_PASSES: AtomicU64 = AtomicU64::new(0);
pub static BG_ABORTED: AtomicU64 = AtomicU64::new(0);
pub static BG_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static BG_CHUNKS: AtomicU64 = AtomicU64::new(0);
pub static BG_US: AtomicU64 = AtomicU64::new(0);
pub static BG_FAILED: AtomicU64 = AtomicU64::new(0);
/// the longest wait one arriving request had behind a background chunk (µs) and the sum
pub static BG_WAIT_MAX_US: AtomicU64 = AtomicU64::new(0);
pub static BG_WAIT_US: AtomicU64 = AtomicU64::new(0);
pub static BG_WAITS: AtomicU64 = AtomicU64::new(0);
pub static HITS: AtomicU64 = AtomicU64::new(0);
pub static HIT_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static MISS_NONE: AtomicU64 = AtomicU64::new(0);
pub static MISS_DIVERGED: AtomicU64 = AtomicU64::new(0);
pub static MISS_NOTDEEPER: AtomicU64 = AtomicU64::new(0);
/// a lane holding a kept state was taken by another conversation (not a miss of this feature: counted apart)
pub static MISS_OTHER: AtomicU64 = AtomicU64::new(0);
/// follow-up admissions (a prompt-end snapshot or a kept state resumed something) and the tokens they prefilled
pub static FOLLOWUPS: AtomicU64 = AtomicU64::new(0);
pub static FOLLOWUP_PREFILLED: AtomicU64 = AtomicU64::new(0);
pub static STATE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static STATE_LANES: AtomicU64 = AtomicU64::new(0);
/// 0 = off, 1 = on, 2 = asked for but fell back (memory / TP / no prefix cache)
pub static MODE_RESUME: AtomicU64 = AtomicU64::new(0);
pub static MODE_NTP: AtomicU64 = AtomicU64::new(0);

pub fn wait_sample(us: u64) {
    BG_WAITS.fetch_add(1, Relaxed);
    BG_WAIT_US.fetch_add(us, Relaxed);
    BG_WAIT_MAX_US.fetch_max(us, Relaxed);
}

pub fn render_metrics(o: &mut String) {
    let g = |a: &AtomicU64| a.load(Relaxed);
    o.push_str("# HELP velogb10_turn TURNRESUME (--turn-resume / --next-turn-prefill): mode_* 0 off, 1 on, 2 asked for but \
                fell back (one boot line says why); state_bytes = device memory of the kept next-turn states; hints = rendered \
                next-turn prefixes the server posted after a reply (dropped = queue overflow, nolane = no idle lane holds that \
                conversation any more, mismatch = the rendering does not start with the lane's prompt); bg_* = background \
                prefill passes (aborted = a request arrived and the pass yielded after its current chunk; wait_us = how long \
                such a request waited for that chunk); hits = admissions resumed from a kept state (hit_tokens = rows the \
                request did not prefill because of it); misses by reason (diverged = the same conversation sent something else than the \
                kept reply: an edit, a branch, a client that renders differently; other_conversation = another conversation took the \
                lane); followups = admissions that resumed any cached \
                prefix, followup_prefilled_tokens = what they still had to prefill\n");
    o.push_str("# TYPE velogb10_turn_hits_total counter\n");
    o.push_str(&format!("velogb10_turn_mode_resume {}\n", g(&MODE_RESUME)));
    o.push_str(&format!("velogb10_turn_mode_next_turn_prefill {}\n", g(&MODE_NTP)));
    o.push_str(&format!("velogb10_turn_state_bytes {}\n", g(&STATE_BYTES)));
    o.push_str(&format!("velogb10_turn_state_lanes {}\n", g(&STATE_LANES)));
    o.push_str(&format!("velogb10_turn_hints_posted_total {}\n", g(&HINTS_POSTED)));
    o.push_str(&format!("velogb10_turn_hints_dropped_total {}\n", g(&HINTS_DROPPED)));
    o.push_str(&format!("velogb10_turn_hints_nolane_total {}\n", g(&HINT_NOLANE)));
    o.push_str(&format!("velogb10_turn_hints_mismatch_total {}\n", g(&HINT_MISMATCH)));
    o.push_str(&format!("velogb10_turn_bg_passes_total {}\n", g(&BG_PASSES)));
    o.push_str(&format!("velogb10_turn_bg_aborted_total {}\n", g(&BG_ABORTED)));
    o.push_str(&format!("velogb10_turn_bg_failed_total {}\n", g(&BG_FAILED)));
    o.push_str(&format!("velogb10_turn_bg_tokens_total {}\n", g(&BG_TOKENS)));
    o.push_str(&format!("velogb10_turn_bg_chunks_total {}\n", g(&BG_CHUNKS)));
    o.push_str(&format!("velogb10_turn_bg_us_total {}\n", g(&BG_US)));
    o.push_str(&format!("velogb10_turn_bg_waits_total {}\n", g(&BG_WAITS)));
    o.push_str(&format!("velogb10_turn_bg_wait_us_total {}\n", g(&BG_WAIT_US)));
    o.push_str(&format!("velogb10_turn_bg_wait_max_us {}\n", g(&BG_WAIT_MAX_US)));
    o.push_str(&format!("velogb10_turn_hits_total {}\n", g(&HITS)));
    o.push_str(&format!("velogb10_turn_hit_tokens_total {}\n", g(&HIT_TOKENS)));
    o.push_str(&format!("velogb10_turn_misses_total{{reason=\"none\"}} {}\n", g(&MISS_NONE)));
    o.push_str(&format!("velogb10_turn_misses_total{{reason=\"diverged\"}} {}\n", g(&MISS_DIVERGED)));
    o.push_str(&format!("velogb10_turn_misses_total{{reason=\"not_deeper\"}} {}\n", g(&MISS_NOTDEEPER)));
    o.push_str(&format!("velogb10_turn_misses_total{{reason=\"other_conversation\"}} {}\n", g(&MISS_OTHER)));
    o.push_str(&format!("velogb10_turn_followups_total {}\n", g(&FOLLOWUPS)));
    o.push_str(&format!("velogb10_turn_followup_prefilled_tokens_total {}\n", g(&FOLLOWUP_PREFILLED)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_for_takes_the_longest_prefix() {
        let h = [1u32, 2, 3, 4, 5, 6];
        let a = [1u32, 2, 3];
        let b = [1u32, 2, 3, 4];
        let c = [1u32, 9];
        assert_eq!(lane_for(&h, &[Some(&a), Some(&b), Some(&c), None]), Some(1));
        assert_eq!(lane_for(&h, &[Some(&c), None]), None);
        // a key longer than the hint never matches; an empty key never matches
        let long = [1u32, 2, 3, 4, 5, 6, 7];
        assert_eq!(lane_for(&h, &[Some(&long), Some(&[][..])]), None);
        // ties: the lowest slot
        assert_eq!(lane_for(&h, &[None, Some(&a), Some(&a)]), Some(1));
    }

    #[test]
    fn pick_rules() {
        let key = [1u32, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(pick(None, &key, 7, 3), Pick::None);
        assert_eq!(pick(Some(&[1, 2, 3, 4, 5]), &key, 7, 3), Pick::Use(5));
        // edited reply: the prompt leaves the kept state
        assert_eq!(pick(Some(&[1, 2, 3, 9, 5]), &key, 7, 3), Pick::Diverged);
        // the prompt is shorter than the kept state
        assert_eq!(pick(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9]), &key, 7, 3), Pick::Diverged);
        // not deeper than the snapshot / past the prefill end (the last prompt token is the seam decode step)
        assert_eq!(pick(Some(&[1, 2, 3]), &key, 7, 3), Pick::NotDeeper);
        assert_eq!(pick(Some(&key[..]), &key, 7, 3), Pick::NotDeeper);
        assert_eq!(pick(Some(&key[..7]), &key, 7, 3), Pick::Use(7));
    }

    #[test]
    fn bg_grid_chunks() {
        assert_eq!(bg_grid(10, 10, 4), Vec::<usize>::new());
        assert_eq!(bg_grid(10, 21, 4), vec![14, 18, 21]);
        assert_eq!(bg_grid(0, 3, 512), vec![3]);
    }

    #[test]
    fn modes() {
        assert_eq!(NtMode::parse("on"), NtMode::On);
        assert_eq!(NtMode::parse("inline"), NtMode::Inline);
        assert_eq!(NtMode::parse("off"), NtMode::Off);
        assert_eq!(NtMode::parse("x"), NtMode::Off);
    }
}
