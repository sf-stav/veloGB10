//! The prefix cache and prefix pool of the dsv41 server (`--prefix-cache on`, `--prefix-pool-gb`).
//!
//! What a kept state holds: everything the forward needs to continue a lane exactly (main compressed KV, index K,
//! all 43 SWA rings, the compressor frontiers, the DSpark carry; the forward's `Snapshot`), plus the token ids it covers.
//! **No Engram n-gram state**: the windows are rebuilt from the token ids (DeepSeek's engine keeps none either, X-DI §3), so a
//! kept state is complete with its tokens, and a hit is EXACT (plan CE: "prefix resume identical to cold").
//!
//! Hit policy (DeepSeek's shape, XREF_DEEPINFER §3):
//! * states live on 128-token blocks near the ends of a prompt (`snapshot_plan`) and at the prompt end and the generation
//!   end of every request; the hit is the longest kept state whose tokens equal the new prompt's first `pos` tokens;
//! * the hit is capped at `L - max(keep, 134)` for a prompt of length `L` (128 window + 6 backtracking rows, so the
//!   rows a resumed prefill recomputes always cover the sliding window);
//! * a prompt that shares more than a kept state but less than the next one backs off to the kept state at or before it
//!   and recomputes the rest exactly (nothing is approximated; the approximate encoder replay is an option for later);
//! * a hit shorter than one block is not taken.

use std::collections::HashMap;

use super::forward::Snapshot;
use crate::dsv41::vision::preproc::ImageSpanKey;

/// Image spans overlapping `0..pos`, with `end` clipped to `pos`.
pub fn spans_covering(spans: &[ImageSpanKey], pos: usize) -> Vec<ImageSpanKey> {
    spans
        .iter()
        .filter(|s| s.start < pos)
        .map(|s| ImageSpanKey { start: s.start, end: s.end.min(pos), digest: s.digest })
        .collect()
}

/// True if `pos` falls strictly inside any image span `(s.start, s.end)`.
pub fn inside_image_span(spans: &[ImageSpanKey], pos: usize) -> bool {
    spans.iter().any(|s| s.start < pos && pos < s.end)
}

/// The store granularity: the compressor blocks and DeepSeek's `KVSTORE_GRAIN`.
pub const BLOCK: usize = 128;
/// Window 128 + 6 backtracking rows: the least a resumed prefill recomputes.
pub const MIN_KEEP: usize = 134;

/// Longest hit allowed for a prompt of `prompt_len` tokens.
pub fn hit_cap(prompt_len: usize, keep: usize) -> usize { prompt_len.saturating_sub(keep.max(MIN_KEEP)) }

/// Positions at which this request's prefill takes a snapshot, ascending, each in `(hit, prompt_len]`:
/// the last `grid_snaps` multiples of 128 that are at or below `prompt_len - keep` (a follow-up can use them whatever
/// its own length is, as long as it is at least `keep` longer), plus the prompt end itself (an identical prefix of a
/// longer prompt). The generation end is added by the engine when the request finishes.
pub fn snapshot_plan(prompt_len: usize, hit: usize, grid_snaps: usize, keep: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let top = hit_cap(prompt_len, keep) / BLOCK * BLOCK;
    let mut p = top;
    for _ in 0..grid_snaps {
        if p == 0 || p <= hit { break; }
        out.push(p);
        p -= BLOCK;
    }
    out.reverse();
    if prompt_len > hit && !out.contains(&prompt_len) { out.push(prompt_len); }
    out
}

/// `rule` (leg PA1): the attention split rule the state was computed under (`attn_rule()`): a state is only resumed under the
/// rule that made it (the two rules give different bits, so their states must never mix).
/// `image_spans` (leg VIS1): content digests and ranges of any image spans inside `0..snap.pos` so two distinct images with
/// identical expanded token counts (`129264`) never collide in the prefix pool.
struct Entry { id: u64, tokens: Vec<u32>, image_spans: Vec<ImageSpanKey>, snap: Snapshot, created: u64, used: u64, rule: u8 }

/// The attention split rule of this process (`--dsv41-attn-split`: 0 = exact, 1 = recipe); part of a kept state's key.
pub fn attn_rule() -> u8 { (crate::opts::lookup("dsv41-attn-split").and_then(crate::opts::get).as_deref() == Some("recipe")) as u8 }

/// Leg CE3: rows a `tail` state may be rolled back to resume a prompt that shares all but its last few tokens (the forward's
/// rollback limit: 6 backtracking rows).
pub const TAIL_BACK: usize = 6;

/// Leg PFX2: minimum prompt-end position for a `tail` state (enables identical / near-identical cache hits on short prompts
/// of `16..128` tokens while keeping `BLOCK` = 128 for all non-tail states).
pub const TAIL_MIN_POS: usize = 16;

/// Leg CE3 / PFX2 / VIS1: where a `tail` state (a full prompt-end state that carries its last rows' taps) resumes `prompt`, if it can:
/// the prompt is at least as long as the state (so the taps the resumed prefill needs below the resume point are the state's own),
/// shares at least `pos - TAIL_BACK` of its tokens, does not split an image span, and matches all image spans covering `0..at`.
fn tail_resume(e: &Entry, prompt: &[u32], spans: &[ImageSpanKey]) -> Option<usize> {
    let pos = e.snap.pos;
    if !e.snap.tail || pos < TAIL_MIN_POS || prompt.len() < pos { return None; }
    let shared = e.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    let at = shared.min(pos).min(prompt.len() - 1);
    (at + TAIL_BACK >= pos
        && at >= TAIL_MIN_POS.saturating_sub(TAIL_BACK)
        && !inside_image_span(&e.image_spans, at)
        && !inside_image_span(spans, at)
        && spans_covering(&e.image_spans, at) == spans_covering(spans, at))
        .then_some(at)
}

/// Leg CE3: the cold-prompt snapshot plan of a forward whose mid-prompt states are encoder-only (`Forward::encoder_cone`): only
/// at the ends of the prefill calls the lane loop makes anyway (`chunk` rows from `hit`; the last call takes a short remainder
/// along when it is under `tail` rows), so the plan adds no call and no row. A call end `b` is kept when a prompt can resume
/// from it at all (`b <= prompt_len - cone`, `b >= 128`) and it is one of: the first 8 call ends (a shared system prompt and
/// tool catalogue), a power-of-two number of calls from `hit`, or the highest usable one (a repeat or an edit near the end);
/// plus the prompt end (a full state: agent follow-ups).
pub fn boundary_plan(prompt_len: usize, hit: usize, chunk: usize, tail: usize, cone: usize) -> Vec<usize> {
    let chunk = chunk.max(1);
    let mut ends = Vec::new();
    let mut e = hit;
    while e < prompt_len {
        let mut end = (e + chunk).min(prompt_len);
        if end < prompt_len && prompt_len - end < tail { end = prompt_len; }
        ends.push(end);
        e = end;
    }
    let usable: Vec<(usize, usize)> = ends.iter().enumerate().map(|(i, &b)| (i + 1, b))
        .filter(|&(_, b)| b < prompt_len && b >= BLOCK && b + cone <= prompt_len).collect();
    let top = usable.last().map(|x| x.1);
    let mut out: Vec<usize> = usable.iter().filter(|&&(n, b)| n <= 8 || n.is_power_of_two() || Some(b) == top).map(|x| x.1).collect();
    if prompt_len > hit { out.push(prompt_len); }
    out
}

/// The pool: kept states under a byte budget, two-tier LRU (`tail == false` evicted before `tail == true`).
pub struct PrefixPool {
    budget: u64,
    keep: usize,
    entries: Vec<Entry>,
    bytes: u64,
    tail_bytes: u64,
    clock: u64,
    next_id: u64,
    pub hits: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub evictions: u64,
    pub tail_evictions: u64,
    pub inserts: u64,
    /// leg CE2: hits the forward could not resume from (`Forward::can_resume`), taken back from `hits` / `reused_tokens`
    pub refused: u64,
}

impl PrefixPool {
    pub fn new(budget_bytes: u64, keep: usize) -> Self {
        PrefixPool { budget: budget_bytes, keep: keep.max(MIN_KEEP), entries: Vec::new(), bytes: 0, tail_bytes: 0, clock: 0, next_id: 1,
                     hits: 0, misses: 0, reused_tokens: 0, evictions: 0, tail_evictions: 0, inserts: 0, refused: 0 }
    }
    pub fn bytes(&self) -> u64 { self.bytes }
    pub fn tail_bytes(&self) -> u64 { self.tail_bytes }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn budget(&self) -> u64 { self.budget }
    pub fn keep(&self) -> usize { self.keep }

    /// The best state for `prompt`: (entry id, covered tokens). Counts a hit or a miss and touches the entry.
    pub fn lookup(&mut self, prompt: &[u32]) -> Option<(u64, usize)> { self.lookup_skip(prompt, &[]) }

    /// `lookup` ignoring the entries in `skip` (leg CE2: states the forward refused for this prompt). The position returned is
    /// where the prompt resumes: the state's own position, or (leg CE3, a `tail` state) up to `TAIL_BACK` rows before it, which
    /// the caller reaches with `Forward::truncate` after the restore.
    pub fn lookup_skip(&mut self, prompt: &[u32], skip: &[u64]) -> Option<(u64, usize)> {
        self.lookup_skip_vl(prompt, &[], skip)
    }

    /// `lookup_skip` matching both token ids and image-span SHA-256 content keys (`spans`).
    pub fn lookup_skip_vl(&mut self, prompt: &[u32], spans: &[ImageSpanKey], skip: &[u64]) -> Option<(u64, usize)> {
        let cap = hit_cap(prompt.len(), self.keep);
        let rule = attn_rule();   // leg PA1: a state is resumed only under the split rule that made it
        let mut best: Option<(usize, usize)> = None; // (entry index, resume position)
        for (i, e) in self.entries.iter().enumerate() {
            if skip.contains(&e.id) { continue; }
            let pos = e.snap.pos;
            // the entry's own floor: an encoder-only state (CED mid-prompt) must leave its whole decoder cone
            let own_cap = prompt.len().saturating_sub(e.snap.min_suffix.max(self.keep));
            if e.rule != rule { continue; }
            let at = if pos >= BLOCK && pos <= cap && pos <= own_cap && e.tokens[..] == prompt[..pos]
                && spans_covering(&e.image_spans, pos) == spans_covering(spans, pos)
            {
                Some(pos)
            } else {
                tail_resume(e, prompt, spans)
            };
            let Some(at) = at else { continue };
            if best.map_or(false, |(_, b)| at <= b) { continue; }
            best = Some((i, at));
        }
        match best {
            Some((i, pos)) => {
                self.clock += 1;
                self.entries[i].used = self.clock;
                self.hits += 1;
                self.reused_tokens += pos as u64;
                Some((self.entries[i].id, pos))
            }
            None => { self.misses += 1; None }
        }
    }

    /// Leg CE2: the hit `(id, pos)` that `lookup` returned could not be used (`Forward::can_resume` refused it): take it back
    /// out of the hit counters and count it as refused; the caller looks up again without it.
    pub fn refuse(&mut self, _id: u64, pos: usize) {
        self.hits = self.hits.saturating_sub(1);
        self.reused_tokens = self.reused_tokens.saturating_sub(pos as u64);
        self.refused += 1;
    }

    pub fn snapshot(&self, id: u64) -> Option<&Snapshot> { self.entries.iter().find(|e| e.id == id).map(|e| &e.snap) }

    fn remove_at(&mut self, i: usize) -> Entry {
        let old = self.entries.remove(i);
        self.bytes -= old.snap.bytes;
        if old.snap.tail { self.tail_bytes -= old.snap.bytes; }
        old
    }

    /// Evict one entry: always `snap.tail == false` (LRU among non-tail) first; `snap.tail == true` (LRU among tail)
    /// only when `allow_tail` is true and zero `snap.tail == false` entries remain.
    fn evict_one(&mut self, allow_tail: bool) -> bool {
        let victim = if allow_tail {
            self.entries.iter().enumerate().min_by_key(|(_, e)| (e.snap.tail, e.used)).map(|(i, _)| i)
        } else {
            self.entries.iter().enumerate().filter(|(_, e)| !e.snap.tail).min_by_key(|(_, e)| e.used).map(|(i, _)| i)
        };
        let Some(i) = victim else { return false };
        let old = self.remove_at(i);
        self.evictions += 1;
        if old.snap.tail { self.tail_evictions += 1; }
        true
    }

    /// Leg PFX2: whether a state of `bytes` fits without evicting any `snap.tail == true` entry when `is_tail == false`,
    /// or fits `self.budget` at all when `is_tail == true`.
    pub fn room_for_kind(&self, bytes: u64, is_tail: bool) -> bool {
        if is_tail { bytes <= self.budget } else { self.tail_bytes.saturating_add(bytes) <= self.budget }
    }

    /// Leg PFX2: pre-evict entries before `Forward::snapshot` allocates on the GPU. Returns false when the state
    /// cannot be admitted (over budget, or non-tail would displace a `tail == true` entry).
    pub fn prepare_insert(&mut self, bytes: u64, is_tail: bool) -> bool {
        if !self.room_for_kind(bytes, is_tail) { return false; }
        while self.bytes.saturating_add(bytes) > self.budget {
            if !self.evict_one(is_tail) { return false; }
        }
        true
    }

    /// Leg PFX2: `prepare_insert` keyed by `tokens` (text-only wrapper for `prepare_insert_for_vl`).
    pub fn prepare_insert_for(&mut self, tokens: &[u32], bytes: u64, is_tail: bool, is_full: bool) -> bool {
        self.prepare_insert_for_vl(tokens, &[], bytes, is_tail, is_full)
    }

    /// Leg PFX2 / VIS1: `prepare_insert` keyed by `(tokens, spans)`: if the pool already holds an equal-or-stronger state
    /// (`tail` covers everything; `Full` covers `Full` and `Encoder`), touches its LRU clock and returns `false` so the
    /// engine skips copying a duplicate snapshot on the GPU; if a weaker state exists, drops it first so its bytes are
    /// reclaimed before checking room or evicting other entries.
    pub fn prepare_insert_for_vl(&mut self, tokens: &[u32], spans: &[ImageSpanKey], bytes: u64, is_tail: bool, is_full: bool) -> bool {
        let pos = tokens.len();
        let min_pos = if is_tail { TAIL_MIN_POS } else { BLOCK };
        if pos < min_pos { return false; }
        let rule = attn_rule();
        let entry_spans = spans_covering(spans, pos);
        if let Some(i) = self.entries.iter().position(|e| {
            e.snap.pos == pos && e.rule == rule && e.tokens[..] == tokens[..] && e.image_spans == entry_spans
        }) {
            let e = &self.entries[i];
            let covers = e.snap.tail || (!is_tail && (!is_full || e.snap.min_suffix == 0));
            if covers {
                self.clock += 1;
                self.entries[i].used = self.clock;
                return false;
            }
            self.remove_at(i);
        }
        self.prepare_insert(bytes, is_tail)
    }

    /// Keep `snap` (covering `tokens[..snap.pos]`) and return its pool entry id when kept.
    pub fn insert_id(&mut self, tokens: &[u32], snap: Snapshot) -> Option<u64> {
        self.insert_id_vl(tokens, &[], snap)
    }

    /// Keep `snap` along with any image spans inside `0..snap.pos` and return its pool entry id when kept.
    pub fn insert_id_vl(&mut self, tokens: &[u32], spans: &[ImageSpanKey], snap: Snapshot) -> Option<u64> {
        let pos = snap.pos;
        let min_pos = if snap.tail { TAIL_MIN_POS } else { BLOCK };
        if pos < min_pos || tokens.len() < pos { return None; }
        let rule = attn_rule();
        let entry_spans = spans_covering(spans, pos);
        if let Some(i) = self.entries.iter().position(|e| {
            e.snap.pos == pos && e.rule == rule && e.tokens[..] == tokens[..pos] && e.image_spans == entry_spans
        }) {
            if self.entries[i].snap.tail && !snap.tail { return None; }
            self.remove_at(i);
        }
        if !self.room_for_kind(snap.bytes, snap.tail) { return None; }
        while self.bytes.saturating_add(snap.bytes) > self.budget {
            if !self.evict_one(snap.tail) { return None; }
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.bytes += snap.bytes;
        if snap.tail { self.tail_bytes += snap.bytes; }
        self.entries.push(Entry { id, tokens: tokens[..pos].to_vec(), image_spans: entry_spans, snap, created: self.clock, used: self.clock, rule });
        self.inserts += 1;
        Some(id)
    }

    /// Keep `snap` (covering `tokens[..snap.pos]`). A state for the same tokens replaces the old one (unless a non-tail
    /// state would overwrite a `tail` state). A state larger than its tier's room is not kept. Returns false when nothing was kept.
    pub fn insert(&mut self, tokens: &[u32], snap: Snapshot) -> bool { self.insert_id_vl(tokens, &[], snap).is_some() }

    /// Keep `snap` along with any image spans inside `0..snap.pos`.
    pub fn insert_vl(&mut self, tokens: &[u32], spans: &[ImageSpanKey], snap: Snapshot) -> bool {
        self.insert_id_vl(tokens, spans, snap).is_some()
    }

    /// Leg PFX2: when `pool.bytes() * 2 > pool.budget()` after a prompt's `tail == true` end snapshot is inserted,
    /// drop this request's un-hit (`used == created`) mid-chunk `tail == false` snapshots (largest `pos` first, keeping
    /// at most `keep_max` early checkpoints only if `bytes * 2 <= budget`). Returns how many entries were dropped.
    pub fn prune_unhit_mid(&mut self, ids: &[u64], keep_max: usize) -> usize {
        if self.bytes.saturating_mul(2) <= self.budget || ids.is_empty() { return 0; }
        let mut cands: Vec<(usize, u64)> = self.entries.iter()
            .filter(|e| !e.snap.tail && e.used == e.created && ids.contains(&e.id))
            .map(|e| (e.snap.pos, e.id))
            .collect();
        cands.sort_unstable_by_key(|&(pos, _)| pos);
        let mut dropped = 0usize;
        while cands.len() > keep_max || (self.bytes.saturating_mul(2) > self.budget && !cands.is_empty()) {
            let (_, id) = cands.pop().unwrap();
            if let Some(i) = self.entries.iter().position(|e| e.id == id) {
                self.remove_at(i);
                dropped += 1;
            }
        }
        dropped
    }

    /// Leg MEM1: change the byte budget (the governor: what the box can spare now). States beyond it are evicted,
    /// non-tail LRU first and `tail` LRU only when no non-tail state remains; returns how many.
    pub fn set_budget(&mut self, budget: u64) -> u64 {
        self.budget = budget;
        let mut n = 0;
        while self.bytes > self.budget {
            if !self.evict_one(true) { break; }
            n += 1;
        }
        n
    }

    /// Would a state of `bytes` fit the budget at all (the engine does not copy one that cannot be kept)?
    pub fn room_for(&self, bytes: u64) -> bool { self.room_for_kind(bytes, true) }

    /// Positions currently kept for a prompt (tests, `/metrics` debug).
    pub fn positions(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.entries.iter().map(|e| e.snap.pos).collect();
        v.sort_unstable();
        v
    }

    pub fn stats(&self) -> HashMap<&'static str, u64> {
        HashMap::from([("hits", self.hits), ("misses", self.misses), ("reused_tokens", self.reused_tokens),
                       ("evictions", self.evictions), ("tail_evictions", self.tail_evictions),
                       ("bytes", self.bytes), ("tail_bytes", self.tail_bytes), ("entries", self.entries.len() as u64)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(pos: usize, bytes: u64) -> Snapshot { Snapshot { pos, bytes, data: Box::new(()), min_suffix: 0, tail: false } }

    #[test]
    fn dsv41_mem1_a_lower_budget_evicts_the_least_recently_used_states() {
        let mut p = PrefixPool::new(1000, 134);
        let t = toks(2000);
        assert!(p.insert(&t, snap(256, 300)));
        assert!(p.insert(&t, snap(512, 300)));
        assert!(p.insert(&t, snap(768, 300)));
        // touch the oldest: the middle one is now the least recently used
        assert_eq!(p.lookup(&t[..256 + 200]).map(|x| x.1), Some(256));
        assert_eq!(p.set_budget(650), 1);
        assert_eq!(p.positions(), vec![256, 768]);
        assert_eq!(p.bytes(), 600);
        assert!(!p.room_for(651) && p.room_for(650));
        // budget 0: everything goes, nothing new is kept
        assert_eq!(p.set_budget(0), 2);
        assert_eq!(p.len(), 0);
        assert!(!p.insert(&t, snap(256, 1)));
        // the box recovered: the budget comes back
        p.set_budget(1000);
        assert!(p.insert(&t, snap(256, 300)));
    }

    #[test]
    fn dsv41_ce_an_encoder_only_state_is_taken_only_when_the_prompt_leaves_its_decoder_cone() {
        let mut p = PrefixPool::new(1 << 30, 134);
        let t = toks(9000);
        // a mid-prompt state of a CED prefill (decoder rings never computed there) and a full state below it
        p.insert(&t, Snapshot { pos: 3072, bytes: 100, data: Box::new(()), min_suffix: 2541, tail: false });
        p.insert(&t, snap(1024, 100));
        // 5,000 tokens: 3072 + 2541 > 5000, so the encoder-only state would need decoder rows it does not hold
        assert_eq!(p.lookup(&t[..5000]).map(|x| x.1), Some(1024));
        // 6,000 tokens leave the cone after it: taken
        assert_eq!(p.lookup(&t[..6000]).map(|x| x.1), Some(3072));
        // the pool's own floor still binds a full state
        p.insert(&t, snap(4096, 100));
        assert_eq!(p.lookup(&t[..4200]).map(|x| x.1), Some(1024));
        assert_eq!(p.lookup(&t[..4230]).map(|x| x.1), Some(4096));
    }
    fn toks(n: usize) -> Vec<u32> { (0..n as u32).map(|i| i * 7 % 1000).collect() }

    #[test]
    fn the_cap_is_prompt_minus_the_window_and_backtracking_rows() {
        assert_eq!(hit_cap(1000, 134), 866);
        assert_eq!(hit_cap(1000, 140), 860);
        assert_eq!(hit_cap(1000, 10), 866, "never below the 128 + 6 minimum");
        assert_eq!(hit_cap(100, 134), 0);
    }

    #[test]
    fn the_plan_is_the_last_grid_blocks_below_the_cap_plus_the_prompt_end() {
        // L = 1000: cap 866 -> grid tops at 768; 4 blocks: 384 512 640 768, then the end
        assert_eq!(snapshot_plan(1000, 0, 4, 134), vec![384, 512, 640, 768, 1000]);
        // resumed at 640: only states beyond the hit are taken
        assert_eq!(snapshot_plan(1000, 640, 4, 134), vec![768, 1000]);
        // short prompt: only the end
        assert_eq!(snapshot_plan(300, 0, 4, 134), vec![128, 300]);
        assert_eq!(snapshot_plan(100, 0, 4, 134), vec![100]);
        assert_eq!(snapshot_plan(1000, 0, 0, 134), vec![1000]);
    }

    #[test]
    fn a_hit_is_the_longest_matching_state_under_the_cap_and_backs_off_between_states() {
        let mut p = PrefixPool::new(1 << 30, 134);
        let t = toks(2000);
        p.insert(&t, snap(256, 100));
        p.insert(&t, snap(512, 100));
        p.insert(&t, snap(1024, 100));
        // a prompt sharing 700 tokens then diverging: the 512 state, not the 1024 one
        let mut q = t[..700].to_vec();
        q.extend((0..600u32).map(|i| 5000 + i));
        assert_eq!(p.lookup(&q).map(|x| x.1), Some(512));
        // a prompt that extends the whole conversation: the 1024 state is allowed up to L - 134
        let long = t[..1200].to_vec();
        assert_eq!(p.lookup(&long).map(|x| x.1), Some(1024));
        // the same 1200 tokens but the cap bites: L = 1100 -> cap 966 -> the 512 state
        assert_eq!(p.lookup(&t[..1100]).map(|x| x.1), Some(512));
        // a diverging prompt shares nothing
        assert_eq!(p.lookup(&(0..900u32).map(|i| 9000 + i).collect::<Vec<_>>()), None);
        assert_eq!((p.hits, p.misses), (3, 1));
        assert_eq!(p.reused_tokens, 512 + 1024 + 512);
    }

    #[test]
    fn states_under_one_block_and_over_the_budget_are_not_kept_and_lru_evicts() {
        let mut p = PrefixPool::new(250, 134);
        let t = toks(2000);
        assert!(!p.insert(&t, snap(64, 10)), "under a block");
        assert!(!p.insert(&t, snap(512, 251)), "over the whole budget");
        assert!(p.insert(&t, snap(256, 100)));
        assert!(p.insert(&t, snap(512, 100)));
        // touch 256 so 512 is the older one... then add a third: the least recently used goes
        assert_eq!(p.lookup(&t[..700]).map(|x| x.1), Some(512));
        assert!(p.insert(&t, snap(768, 100)));
        assert_eq!(p.positions(), vec![512, 768]);
        assert_eq!(p.evictions, 1);
        assert_eq!(p.bytes(), 200);
        // the same tokens replace, they do not duplicate
        assert!(p.insert(&t, snap(768, 120)));
        assert_eq!(p.positions(), vec![512, 768]);
        assert_eq!(p.bytes(), 220);
    }

    fn tail_snap(pos: usize, bytes: u64) -> Snapshot { Snapshot { pos, bytes, data: Box::new(()), min_suffix: 0, tail: true } }

    #[test]
    fn dsv41_pfx2_two_tier_eviction_protects_tail_states_from_non_tail_churn() {
        let mut p = PrefixPool::new(1000, 134);
        let t1 = toks(3000);
        let mut t2 = toks(3000); t2[0] = 9999;
        let mut t3 = toks(3000); t3[0] = 8888;
        // Two prompt-end tail states (300 + 300 = 600 B) and one mid-chunk non-tail state (300 B)
        assert!(p.insert(&t1, tail_snap(256, 300)));
        assert!(p.insert(&t2, tail_snap(512, 300)));
        assert!(p.insert(&t3, snap(768, 300)));
        assert_eq!((p.bytes(), p.tail_bytes(), p.tail_evictions), (900, 600, 0));
        // Touch the non-tail state so it is the MOST recently used overall: a new state must STILL evict it before any tail state
        assert_eq!(p.lookup(&t3[..1000]).map(|x| x.1), Some(768));
        let mut t4 = toks(3000); t4[0] = 7777;
        assert!(p.room_for_kind(400, false));
        assert!(!p.room_for_kind(401, false), "tail_bytes (600) + 401 > budget (1000)");
        assert!(p.prepare_insert(400, false));
        assert_eq!((p.evictions, p.tail_evictions, p.bytes(), p.tail_bytes()), (1, 0, 600, 600));
        assert!(p.insert(&t4, snap(1024, 400)));
        assert_eq!(p.positions(), vec![256, 512, 1024]);
        // A non-tail state that would force evicting a tail state is refused by both prepare_insert and insert
        let mut t5 = toks(3000); t5[0] = 6666;
        assert!(!p.prepare_insert(450, false));
        assert!(!p.insert(&t5, snap(1280, 450)));
        assert_eq!(p.tail_evictions, 0);
        assert_eq!(p.positions(), vec![256, 512, 1024]);
        // A non-tail state at the exact same tokens never downgrades an existing tail state
        assert!(!p.insert(&t1, snap(256, 100)));
        assert!(p.snapshot(1).unwrap().tail);
        // Governor lowering budget to 650 evicts the non-tail 1024 state first, keeping both tail states
        assert_eq!(p.set_budget(650), 1);
        assert_eq!((p.positions(), p.tail_evictions), (vec![256, 512], 0));
        // Only when zero non-tail states remain and a new tail state (or lower budget) demands room does LRU among tail states evict
        assert_eq!(p.lookup(&t1[..256]).map(|x| x.1), Some(255)); // touch t1 so t2 (512) is LRU among tail states
        assert!(p.insert(&t5, tail_snap(384, 300)));
        assert_eq!((p.positions(), p.tail_evictions), (vec![256, 384], 1));
    }

    #[test]
    fn dsv41_pfx2_short_tail_states_down_to_16_tokens_hit_while_non_tail_keeps_128_floor() {
        let mut p = PrefixPool::new(1 << 20, 134);
        let base = toks(500);
        assert!(!p.insert(&base, tail_snap(15, 100)), "below TAIL_MIN_POS (16)");
        assert!(!p.insert(&base, snap(87, 100)), "non-tail below BLOCK (128) still rejected");
        for &len in &[16usize, 31, 46, 47, 68, 79, 87] {
            let mut pr = base[..len].to_vec();
            pr[0] = 10_000 + len as u32;
            assert!(p.insert(&pr, tail_snap(len, 100)), "tail state at {len}");
            // Identical short prompt: resumes at len - 1 (both odd and even `at`, e.g. 15, 30, 45, 46, 67, 78, 86)
            assert_eq!(p.lookup(&pr).map(|x| x.1), Some(len - 1), "identical {len}");
            // Near-identical (last 4 tokens edited): resumes at len - 4 (e.g. 16 -> 12, 46 -> 42, 47 -> 43)
            let mut ed = pr[..len - 4].to_vec();
            ed.extend_from_slice(&[9991, 9992, 9993, 9994]);
            assert_eq!(p.lookup(&ed).map(|x| x.1), Some(len - 4), "edit4 {len}");
            // Last 7 tokens edited: exceeds TAIL_BACK (6) -> miss
            if len > 7 {
                let mut ed7 = pr[..len - 7].to_vec();
                ed7.extend_from_slice(&[9995; 7]);
                assert_eq!(p.lookup(&ed7), None, "edit7 {len}");
            }
            // Short extension (< 128) and long extension (>= 128): resumes at len
            let mut ext_short = pr.clone();
            ext_short.extend_from_slice(&[7777; 12]);
            assert_eq!(p.lookup(&ext_short).map(|x| x.1), Some(len), "ext_short {len}");
            let mut ext_long = pr.clone();
            ext_long.extend_from_slice(&[8888; 200]);
            assert_eq!(p.lookup(&ext_long).map(|x| x.1), Some(len), "ext_long {len}");
        }
    }

    #[test]
    fn dsv41_pfx2_prune_unhit_mid_drops_dead_scaffolding_above_half_budget() {
        let mut p = PrefixPool::new(1000, 134);
        let t = toks(20_000);
        let c1 = p.insert_id(&t, snap(2048, 120)).unwrap();
        let c2 = p.insert_id(&t, snap(4096, 150)).unwrap();
        let c3 = p.insert_id(&t, snap(8192, 200)).unwrap();
        let c4 = p.insert_id(&t, snap(16384, 250)).unwrap();
        // Touch c2 as if another concurrent request hit it
        assert_eq!(p.lookup(&t[..4300]).map(|x| x.1), Some(4096));
        let _end = p.insert_id(&t, tail_snap(18000, 260)).unwrap();
        assert_eq!(p.bytes(), 980); // 980 * 2 > 1000
        let dropped = p.prune_unhit_mid(&[c1, c2, c3, c4], 2);
        assert!(dropped >= 2, "dropped {dropped}");
        assert!(p.bytes() * 2 <= p.budget(), "bytes {} vs budget {}", p.bytes(), p.budget());
        assert!(p.positions().contains(&4096), "hit mid-chunk kept");
        assert!(p.positions().contains(&18000), "tail state kept");
    }

    #[test]
    fn dsv41_pfx2_velobenchmark_3_runs_100_percent_tail_hits_in_656mb_pool() {
        use crate::dsv41::config::{test_config_json, Dsv41Config};
        use crate::dsv41::tp::mem::snap_bytes;
        let cfg = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        let lens: [usize; 16] = [31, 87, 68, 79, 46, 47, 3354, 1174, 2324, 4598, 9173, 18293, 36565, 73082, 146133, 219190];
        let budget: u64 = 656_000_000; // 656 MB TP=2 Rank 0 governed budget
        let mut pool = PrefixPool::new(budget, MIN_KEEP);
        let tail_tap = |pos: usize| (pos.min(128) * cfg.dspark_targets.len() * cfg.hidden * 2) as u64;
        let sbytes = |pos: usize, is_tail: bool| snap_bytes(&cfg, pos) + if is_tail { tail_tap(pos) } else { 0 };
        let prompts: Vec<Vec<u32>> = lens.iter().enumerate().map(|( idx, &n)| {
            (0..n as u32).map(|i| 100 + idx as u32 * 9973 + i.wrapping_mul(2654435761) % 90_000).collect()
        }).collect();

        for run in 1..=3 {
            let mut run_tail_hits = 0usize;
            for (pi, pr) in prompts.iter().enumerate() {
                let plen = pr.len();
                let hit = pool.lookup(pr);
                if run >= 2 {
                    assert_eq!(hit.map(|x| x.1), Some(plen - 1), "run {run} prompt #{pi} (len {plen}) must hit tail_resume");
                    run_tail_hits += 1;
                }
                let cached = hit.map_or(0, |x| x.1);
                let plan = boundary_plan(plen, cached, 2048, 128, 2541);
                let mut mids = Vec::new();
                for &end in &plan {
                    let is_tail = end == plen;
                    let b = sbytes(end, is_tail);
                    if pool.prepare_insert_for(&pr[..end], b, is_tail, is_tail) {
                        assert!(pool.bytes() + b <= budget, "pre-evicted before GPU alloc");
                        let snap = Snapshot { pos: end, bytes: b, data: Box::new(()), min_suffix: if is_tail { 0 } else { 2541 }, tail: is_tail };
                        if let Some(id) = pool.insert_id(&pr[..end], snap) {
                            if !is_tail { mids.push(id); }
                        }
                    }
                    if is_tail && !mids.is_empty() && pool.bytes() * 2 > pool.budget() {
                        pool.prune_unhit_mid(&mids, 2);
                    }
                }
                // Generation end (`end_rounds`: plen + 63 tokens, `tail == false`)
                let gen_end = plen + 63;
                if gen_end >= BLOCK {
                    let mut fed = pr.clone();
                    fed.extend((0..63u32).map(|j| 90_000 + j));
                    let gb = sbytes(gen_end, false);
                    if pool.prepare_insert_for(&fed, gb, false, true) {
                        let _ = pool.insert_id(&fed, Snapshot { pos: gen_end, bytes: gb, data: Box::new(()), min_suffix: 0, tail: false });
                    }
                }
            }
            if run >= 2 {
                assert_eq!(run_tail_hits, lens.len(), "run {run}: 100% tail_resume hits across all 16 prompts");
            }
        }
        assert_eq!(pool.tail_evictions, 0, "zero tail == true evictions across all 3 VeloBenchmark runs");
    }

    #[test]
    fn dsv41_vis1_same_shape_different_images_never_collide_and_followup_turn_hits() {
        let mut p = PrefixPool::new(1 << 30, 134);
        let mut t = toks(800);
        for slot in &mut t[10..199] { *slot = 129_264; }
        let span_a = vec![ImageSpanKey { start: 10, end: 199, digest: [0xAA; 32] }];
        let span_b = vec![ImageSpanKey { start: 10, end: 199, digest: [0xBB; 32] }];
        assert!(p.insert_vl(&t[..512], &span_a, Snapshot { pos: 512, bytes: 100, data: Box::new(()), min_suffix: 0, tail: true }));
        // Different image of the exact same grid (189 tokens of 129264 at 10..199): MUST miss both direct and tail_resume!
        assert_eq!(p.lookup_skip_vl(&t[..512], &span_b, &[]), None, "tail_resume must not collide across distinct images");
        assert_eq!(p.lookup_skip_vl(&t[..700], &span_b, &[]), None, "prefix lookup must not collide across distinct images");
        // Follow-up turn on the SAME image (even when Turn 2 adds a second image after 512): MUST hit 512!
        let mut span_a_turn2 = span_a.clone();
        span_a_turn2.push(ImageSpanKey { start: 550, end: 739, digest: [0xCC; 32] });
        assert_eq!(p.lookup_skip_vl(&t[..800], &span_a_turn2, &[]).map(|x| x.1), Some(512));
        // Inserting state B at the same pos=512 keeps BOTH entries instead of overwriting A!
        assert!(p.insert_vl(&t[..512], &span_b, Snapshot { pos: 512, bytes: 100, data: Box::new(()), min_suffix: 0, tail: true }));
        assert_eq!(p.len(), 2);
        assert_eq!(p.lookup_skip_vl(&t[..700], &span_b, &[]).map(|x| x.1), Some(512));
        assert!(inside_image_span(&span_a, 100) && !inside_image_span(&span_a, 10) && !inside_image_span(&span_a, 199));
    }
}
