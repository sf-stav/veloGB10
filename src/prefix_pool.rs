//! PFXPOOL (2026-11-10): the prefix pool — prompt states kept in memory ACROSS lanes.
//!
//! The prefix cache keeps one prompt-end snapshot per lane, in the lane: a second conversation on the
//! lane overwrites it. With 2-4 lanes and more distinct prompts than lanes (a benchmark's 13 turns, an
//! agent's several conversations) a repeated prompt never finds its state again (owner report
//! 2026-11-10: VeloBench 1_FULL_TEST run twice gains nothing on veloGB10, while the rival's second run
//! reads every prompt from its cache). The pool fixes that the way llama.cpp's host prompt cache and
//! TensorFold's prompt-state pool do: when a lane is taken over, the outgoing conversation's prompt
//! state (the positional rows it uses + the recurrent snapshot, exactly what the spill tier captures)
//! is COPIED into a pool entry in device (= unified) memory; a later prompt that starts with an entry's
//! tokens is restored from it with one device copy instead of being read again.
//!
//! This module is the host-side policy, generic over the buffer so it is unit-tested without a GPU:
//! longest-prefix lookup, exact-key dedupe, a COST-WEIGHTED byte cap (PFXPOOL3). Under TP every rank holds the same policy
//! state: every input to it (keys, token counts, the per-entry cost, the agreed capture outcome) is
//! replicated, so lookups and evictions come out identically on every rank.
//!
//! PFXPOOL2 (2026-11-10, owner hand test: hits stopped, then the box hit the memory watchdog): related prompts
//! (a benchmark's nested 1K, 2K, ... 256K fills; a conversation's turns; prompts on one document) used to cost one
//! full copy of their rows each. Positional rows are now SHARED: every entry carries the LINEAGE of its rows (`Segs`:
//! which prefill computation wrote each row range). A new entry whose leading rows have the same lineage as an older
//! entry's keeps only the rows after that point (`share` rows come from its PARENT, recursively) plus its own
//! recurrent snapshot. Equal lineage = the same bytes (rows sharing a generation id were copied, never recomputed),
//! so a restore through the chain is exactly the restore its own full copy would have been. Evicting an entry
//! evicts its descendants; touching an entry touches its ancestors, so no victim order strands a descendant.
//!
//! PFXPOOL3 (2026-11-22, the 215k-token session in server.log): `make_room` no longer evicts by pure LRU. A
//! 551-token helper capture shifted out a 215k-token session's entry purely by age, and the session's next request
//! re-prefilled 162 s (1326 tok/s measured). Victim order is now ASCENDING re-prefill cost — a family's cost is its
//! deepest entry's token count (the whole family goes together, so that is the re-prefill its loss actually costs),
//! bytes only as the tiebreak, then the LRU stamp, then the id — and a capture whose room can only be made by
//! evicting a family DEEPER than the entrant is REFUSED outright: `make_room` returns None and the pool stays
//! untouched (the entrant is cheaper to re-read than anything it would displace, by definition). The memory-floor
//! paths (`pop_lru`, upkeep, emergency) stay LRU: they free memory to survive, not to admit one entrant.

/// Row lineage: ascending (end, generation) pairs covering rows 0..last end; rows [prev end, end) were written by
/// prefill computation `generation`. Two row sets agree on 0..n exactly when their lineages clipped at n are equal.
pub type Segs = Vec<(usize, u64)>;

/// `s` clipped to rows 0..n (None when it does not cover n rows).
pub fn clip(s: &[(usize, u64)], n: usize) -> Option<Segs> {
    if n == 0 { return Some(Vec::new()); }
    let mut out = Vec::new();
    for &(e, g) in s {
        if e >= n { out.push((n, g)); return Some(out); }
        out.push((e, g));
    }
    None
}

/// Do two lineages agree on rows 0..n?
pub fn same_rows(a: &[(usize, u64)], b: &[(usize, u64)], n: usize) -> bool {
    match (clip(a, n), clip(b, n)) { (Some(x), Some(y)) => x == y, _ => false }
}

/// How many leading rows two lineages agree on.
pub fn common_rows(a: &[(usize, u64)], b: &[(usize, u64)]) -> usize {
    let (mut i, mut j, mut at) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i].1 != b[j].1 { return at; }
        let e = a[i].0.min(b[j].0);
        if a[i].0 != b[j].0 { return e; }
        at = e;
        i += 1;
        j += 1;
    }
    at
}

/// One pool entry: the tokens whose state it holds (rows 0..toks.len() + the recurrent state after
/// them), its cost in bytes (the deterministic per-rank size), the buffer, its LRU stamp.
pub struct Entry<B> {
    pub toks: Vec<u32>,
    pub bytes: u64,
    pub used: u64,
    pub buf: B,
    /// positions of the recurrent checkpoints the entry carries (ascending, all < toks.len()): a prompt that shares
    /// only part of `toks` can resume at the deepest one within the shared part
    pub marks: Vec<usize>,
    /// unique id (replicated: assigned in insert order)
    pub id: u64,
    /// Some(parent id): rows 0..`share` are read from the parent (recursively); this entry's own buffer holds rows
    /// `share`.. only
    pub parent: Option<u64>,
    pub share: usize,
    /// length of the parent chain (0 for an entry that holds all its rows)
    pub depth: usize,
    /// the lineage of rows 0..toks.len()
    pub segs: Segs,
}

/// Where a prompt can resume from the pool: entry `i`, at `at` tokens; `full` = the entry's own prompt-end state
/// (its tokens prefix the prompt), else a carried checkpoint at `at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resume {
    pub i: usize,
    pub at: usize,
    pub full: bool,
}

pub struct Pool<B> {
    cap: u64,
    bytes: u64,
    tick: u64,
    next_id: u64,
    entries: Vec<Entry<B>>,
}

impl<B> Pool<B> {
    pub fn new(cap_bytes: u64) -> Self { Self { cap: cap_bytes, bytes: 0, tick: 0, next_id: 1, entries: Vec::new() } }
    pub fn cap_bytes(&self) -> u64 { self.cap }
    pub fn bytes(&self) -> u64 { self.bytes }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn on(&self) -> bool { self.cap > 0 }

    /// Is there an entry for exactly these tokens? (capture dedupe)
    pub fn contains(&self, toks: &[u32]) -> bool { self.entries.iter().any(|e| e.toks[..] == *toks) }

    /// The deepest entry this prompt can resume from: its tokens are a prefix of `prompt` and it holds
    /// at most `end` tokens (the prefill end, plen-1: the last prompt token is the seam step). Ties
    /// (equal length = equal tokens, impossible after dedupe) go to the first. Returns (index, tokens).
    pub fn lookup(&self, prompt: &[u32], end: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.toks.len();
            if n == 0 || n > end || n > prompt.len() || prompt[..n] != e.toks[..] { continue; }
            if best.map_or(true, |(_, b)| n > b) { best = Some((i, n)); }
        }
        best
    }

    /// The deepest resume this prompt has in the pool: an entry whose tokens prefix it (at most `end` of them), or a
    /// carried checkpoint at most min(LCP, end) deep. Full beats a checkpoint at the same depth.
    pub fn best(&self, prompt: &[u32], end: usize) -> Option<Resume> {
        let mut best: Option<Resume> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.toks.len();
            let lcp = e.toks.iter().zip(prompt).take_while(|(a, b)| a == b).count();
            let cand = if lcp == n && n > 0 && n <= end {
                Some(Resume { i, at: n, full: true })
            } else {
                let lim = lcp.min(end);
                e.marks.iter().rev().find(|&&m| m > 0 && m <= lim).map(|&m| Resume { i, at: m, full: false })
            };
            if let Some(c) = cand {
                if best.map_or(true, |b| c.at > b.at || (c.at == b.at && c.full && !b.full)) { best = Some(c); }
            }
        }
        best
    }

    pub fn get(&self, i: usize) -> Option<&Entry<B>> { self.entries.get(i) }
    pub fn get_mut(&mut self, i: usize) -> Option<&mut Entry<B>> { self.entries.get_mut(i) }

    /// Entry indices in the order repeated `pop_lru` calls would take them (each root of a popped family first).
    pub fn lru_order(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.entries.len()).collect();
        idx.sort_by_key(|&i| self.entries[i].used); // stable: equal stamps keep entry order, as min_by_key does
        idx
    }

    /// Indices of entry `i` and its descendants.
    pub fn family_of(&self, i: usize) -> Vec<usize> {
        self.entries.get(i).map_or(Vec::new(), |e| self.family(e.id).into_iter().filter_map(|id| self.index_of(id)).collect())
    }

    pub fn index_of(&self, id: u64) -> Option<usize> { self.entries.iter().position(|e| e.id == id) }

    /// Entry `i` and its ancestors (ids, nearest first).
    pub fn chain(&self, i: usize) -> Vec<u64> {
        let mut out = Vec::new();
        let mut cur = self.entries.get(i);
        while let Some(e) = cur {
            out.push(e.id);
            cur = e.parent.and_then(|p| self.entries.iter().find(|r| r.id == p));
        }
        out
    }

    /// Where rows lo..hi of entry `i` live: (entry index, from, to) parts, deepest ancestor last. Each entry holds rows
    /// share..its end itself.
    pub fn row_parts(&self, i: usize, hi: usize) -> Option<Vec<(usize, usize, usize)>> {
        let mut out = Vec::new();
        let mut cur = i;
        let mut hi = hi;
        loop {
            let e = self.entries.get(cur)?;
            let lo = if e.parent.is_some() { e.share.min(hi) } else { 0 };
            if hi > lo { out.push((cur, lo, hi)); }
            if lo == 0 { return Some(out); }
            hi = lo;
            cur = self.index_of(e.parent?)?;
        }
    }

    /// LRU: a restore from entry `i` counts as a use (of its ancestors too).
    pub fn touch(&mut self, i: usize) {
        self.tick += 1;
        let t = self.tick;
        for id in self.chain(i) {
            if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) { e.used = t; }
        }
    }

    /// The entry a new capture with row lineage `segs` shares the most leading rows with: (index, rows in common).
    /// Only entries whose chain is shorter than `max_depth`; first on ties (deterministic).
    pub fn best_parent(&self, segs: &[(usize, u64)], max_depth: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.depth + 1 > max_depth { continue; }
            let k = common_rows(&e.segs, segs).min(e.toks.len());
            if k > 0 && best.map_or(true, |(_, b)| k > b) { best = Some((i, k)); }
        }
        best
    }

    /// Would an entry of `bytes` ever fit under the cap?
    pub fn fits(&self, bytes: u64) -> bool { bytes > 0 && bytes <= self.cap }

    /// PFXPOOL3: free room for `bytes` under the cap by re-prefill cost: victim families come out in ASCENDING
    /// cost order — a family's cost is its DEEPEST entry's token count (the family goes together, so that is the
    /// re-prefill losing it actually costs), then total family bytes, then the family's oldest LRU stamp, then the
    /// root id — and never a family deeper than the entrant (`entrant_toks` = the tokens of the capture being
    /// admitted). Returns Some(evicted) (the caller frees the buffers or hands them to the SSD tier), or None =
    /// REFUSED: the families shallow enough to be evictable cannot free enough, so the newcomer is not kept and
    /// the pool is untouched (refusal moves nothing — the plan is built before any removal). Deterministic:
    /// entries, ids and stamps are replicated state, so every rank computes the same victim set or the same
    /// refusal from the same inputs. The byte cap of the memory-FLOOR paths stays LRU (`pop_lru`): that pressure
    /// is not caused by an entrant, so cost-weighting has no entrant to compare against. A family's cost counts
    /// only members not already claimed by an EARLIER (cheaper) pick — the descendant family chosen first frees
    /// its bytes, and the ancestor root then costs only what remains of its subtree.
    pub fn make_room(&mut self, bytes: u64, entrant_toks: usize) -> Option<Vec<Entry<B>>> {
        if self.bytes.saturating_add(bytes) <= self.cap {
            return Some(Vec::new());
        }
        let mut need = self.bytes + bytes - self.cap;
        // the plan runs on the untouched pool: each entry as a candidate subtree root -> (root id, members
        // (tokens, bytes, used, id)); a member already claimed by a chosen victim contributes no cost again
        let fams: Vec<(u64, Vec<(usize, u64, u64, u64)>)> = self.entries.iter().map(|e| {
            (e.id, self.family(e.id).into_iter().filter_map(|fid| self.entries.iter().find(|x| x.id == fid)
                .map(|x| (x.toks.len(), x.bytes, x.used, x.id))).collect())
        }).collect();
        let mut gone: Vec<u64> = Vec::new(); // ids claimed by a family already chosen
        let mut roots: Vec<u64> = Vec::new();
        loop {
            let best = fams.iter().filter_map(|(rid, f)| {
                let (mut d, mut b, mut u) = (0usize, 0u64, u64::MAX);
                for (tk, by, us, id) in f {
                    if gone.contains(id) {
                        continue; // already freed by a cheaper descendant family
                    }
                    d = d.max(*tk);
                    b += *by;
                    u = u.min(*us);
                }
                (b > 0 && d <= entrant_toks).then_some((d, b, u, *rid))
            }).min_by_key(|(d, b, u, rid)| (*d, *b, *u, *rid));
            let Some((_, b, _, rid)) = best else { return None }; // nothing eligible frees enough: refuse
            need = need.saturating_sub(b);
            roots.push(rid);
            if let Some((_, f)) = fams.iter().find(|(r, _)| *r == rid) {
                gone.extend(f.iter().map(|(_, _, _, id)| *id));
            }
            if need == 0 {
                break;
            }
        }
        let mut out = Vec::new();
        for rid in roots {
            if let Some(i) = self.index_of(rid) {
                out.extend(self.remove_at(i));
            }
        }
        Some(out)
    }

    /// Evict the least recently used entry, with its children when it is a root (empty when the pool is).
    pub fn pop_lru(&mut self) -> Vec<Entry<B>> {
        if self.entries.is_empty() { Vec::new() } else { self.evict_lru() }
    }

    /// The LRU entry's id.
    pub fn lru(&self) -> Option<u64> { self.entries.iter().min_by_key(|e| e.used).map(|e| e.id) }

    /// Ids of entry `id` and all its descendants.
    fn family(&self, id: u64) -> Vec<u64> {
        let mut out = vec![id];
        let mut k = 0;
        while k < out.len() {
            let p = out[k];
            for e in &self.entries { if e.parent == Some(p) { out.push(e.id); } }
            k += 1;
        }
        out
    }

    /// The bytes evicting the LRU entry would free (with its descendants).
    pub fn lru_bytes(&self) -> u64 {
        let Some(id) = self.lru() else { return 0 };
        let f = self.family(id);
        self.entries.iter().filter(|e| f.contains(&e.id)).map(|e| e.bytes).sum()
    }

    fn evict_lru(&mut self) -> Vec<Entry<B>> {
        let i = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i).unwrap();
        self.remove_at(i)
    }

    /// Remove entry `id` and its descendants (empty when it is gone already).
    pub fn remove_id(&mut self, id: u64) -> Vec<Entry<B>> {
        match self.index_of(id) { Some(i) => self.remove_at(i), None => Vec::new() }
    }

    /// Ids of every entry, ascending (a replicated order).
    pub fn ids(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.entries.iter().map(|e| e.id).collect();
        v.sort_unstable();
        v
    }

    /// Remove entry `i` and its descendants (the entry first).
    fn remove_at(&mut self, i: usize) -> Vec<Entry<B>> {
        let f = self.family(self.entries[i].id);
        let mut out = Vec::new();
        for id in f {
            if let Some(k) = self.index_of(id) {
                let e = self.entries.swap_remove(k);
                self.bytes -= e.bytes;
                out.push(e);
            }
        }
        out
    }

    /// Insert an entry (the caller ran `make_room` first). An entry with the same tokens is replaced
    /// and returned.
    pub fn insert(&mut self, toks: Vec<u32>, bytes: u64, buf: B) -> Vec<Entry<B>> {
        let segs = vec![(toks.len(), 0)];
        self.insert_marked(toks, bytes, buf, Vec::new(), None, segs).1
    }

    /// `insert` with the positions of the checkpoints the entry carries, its parent (id, shared rows; None = it holds
    /// all its rows) and its row lineage. Returns the new id and whatever it replaced (an entry with the same tokens,
    /// with its descendants).
    pub fn insert_marked(&mut self, toks: Vec<u32>, bytes: u64, buf: B, marks: Vec<usize>, parent: Option<(u64, usize)>,
                         segs: Segs) -> (u64, Vec<Entry<B>>) {
        let old = match self.entries.iter().position(|e| e.toks == toks) {
            Some(i) => self.remove_at(i),
            None => Vec::new(),
        };
        let (parent, share, depth) = match parent.and_then(|(p, k)| self.index_of(p).map(|i| (p, k, self.entries[i].depth))) {
            Some((p, k, d)) => (Some(p), k, d + 1),
            None => (None, 0, 0),
        };
        let id = self.next_id;
        self.next_id += 1;
        self.bytes += bytes;
        self.entries.push(Entry { toks, bytes, used: 0, buf, marks, id, parent, share, depth, segs });
        let i = self.entries.len() - 1;
        self.touch(i);
        (id, old)
    }

    /// Remove every entry (e.g. a fatal restore error under TP is reported by the caller; tests).
    pub fn clear(&mut self) -> Vec<Entry<B>> {
        self.bytes = 0;
        std::mem::take(&mut self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: usize, base: u32) -> Vec<u32> { (0..n as u32).map(|i| base + i).collect() }

    #[test]
    fn lookup_takes_the_deepest_prefix_within_end() {
        let mut p: Pool<u8> = Pool::new(1000);
        p.insert(t(10, 0), 10, 1);
        p.insert(t(20, 0), 20, 2);
        p.insert(t(30, 0), 30, 3);
        let prompt = t(31, 0);
        // end = 30: the 30-token entry is usable
        assert_eq!(p.lookup(&prompt, 30), Some((2, 30)));
        // end = 25: only up to 20
        assert_eq!(p.lookup(&prompt, 25).map(|x| x.1), Some(20));
        // a prompt that diverges at 15: only the 10-token entry is a prefix
        let mut q = t(31, 0);
        q[15] = 999;
        assert_eq!(p.lookup(&q, 30).map(|x| x.1), Some(10));
        // nothing matches a different prompt
        assert_eq!(p.lookup(&t(31, 100), 30), None);
    }

    #[test]
    fn exact_repeat_hits_its_own_entry() {
        // a prompt-end entry holds plen-1 tokens; the same prompt again has end = plen-1
        let mut p: Pool<u8> = Pool::new(1000);
        let prompt = t(50, 7);
        p.insert(prompt[..49].to_vec(), 49, 1);
        assert_eq!(p.lookup(&prompt, 49), Some((0, 49)));
        assert!(p.contains(&prompt[..49]));
    }

    #[test]
    fn lru_cap_evicts_oldest_unused_first() {
        let mut p: Pool<u8> = Pool::new(100);
        p.insert(t(1, 0), 40, 1);
        p.insert(t(1, 10), 40, 2);
        p.touch(0); // entry 0 used again: entry 1 is now the LRU
        // PFXPOOL3: equal cost (1 token, 40 bytes each) — the LRU-stamp tiebreak shifts out the older one first
        let ev = p.make_room(40, 1).unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].buf, 2);
        p.insert(t(1, 20), 40, 3);
        assert_eq!(p.bytes(), 80);
        assert!(p.bytes() <= p.cap_bytes());
        assert!(!p.fits(101));
        assert!(p.fits(100));
    }

    #[test]
    fn cost_weighted_make_room_refuses_the_cheaper_entrant() {
        // PFXPOOL3 acceptance (b): a shallower entry can NEVER evict a deeper one; a deeper entrant evicts the
        // shallower first; equal depth may displace equal depth; a refusal leaves the pool untouched.
        let mut p: Pool<u8> = Pool::new(100);
        p.insert(t(50, 0), 40, 1);   // the session: 50 tokens
        p.insert(t(60, 100), 40, 2); // its longer turn: 60 tokens
        assert_eq!(p.bytes(), 80);
        // a 10-token entrant needs 20 bytes of room — everything here is deeper than it: refused, untouched
        assert!(p.make_room(40, 10).is_none());
        assert_eq!(p.len(), 2);
        assert_eq!(p.bytes(), 80);
        // equal depth is not deeper: a 60-token entrant takes the SHALLOWEST eligible family first (the 50 one)
        let ev = p.make_room(40, 60).unwrap();
        assert_eq!(ev.iter().map(|e| e.buf).collect::<Vec<_>>(), vec![1]);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn cost_weighted_make_room_evicts_ascending_and_takes_families_together() {
        // PFXPOOL3: a family's cost is its deepest token count; a child alone is the cheaper family; taking the
        // root takes the descendants; a chain deeper than the entrant is untouchable (refusal, pool untouched).
        let mut p: Pool<u8> = Pool::new(8000);
        let (a, _) = p.insert_marked(t(4000, 0), 4000, 10, vec![], None, vec![(4000, 1)]);
        p.insert_marked(t(8000, 0), 3900, 20, vec![], Some((a, 4000)), vec![(4000, 1), (8000, 2)]);
        assert_eq!(p.bytes(), 7900);
        // a deeper entrant: the child alone (8000 tok / 3900 B) is cheaper than the root's family
        // (8000 tok / 7900 B) — the child goes, the parent stays
        let ev = p.make_room(4000, 9000).unwrap();
        assert_eq!(ev.iter().map(|e| e.buf).collect::<Vec<_>>(), vec![20]);
        assert_eq!(p.len(), 1);
        assert!(p.index_of(a).is_some());
        // the same chain against a 5000-token entrant: both families are deeper than it -> refused untouched
        let mut q: Pool<u8> = Pool::new(8000);
        let (qa, _) = q.insert_marked(t(4000, 0), 4000, 10, vec![], None, vec![(4000, 1)]);
        q.insert_marked(t(8000, 0), 3900, 20, vec![], Some((qa, 4000)), vec![(4000, 1), (8000, 2)]);
        assert!(q.make_room(4000, 5000).is_none());
        assert_eq!(q.len(), 2);
        assert_eq!(q.bytes(), 7900);
        // a deep entrant under real pressure: leaves go ascending — the child first, then the root's family
        let ev = q.make_room(7000, 9000).unwrap(); // free 6900: child (3900) then the root (4000)
        assert_eq!(ev.iter().map(|e| e.buf).collect::<Vec<_>>(), vec![20, 10]);
        assert!(q.is_empty());
    }

    #[test]
    fn insert_same_tokens_replaces_and_accounts() {
        let mut p: Pool<u8> = Pool::new(100);
        p.insert(t(5, 0), 30, 1);
        let old = p.insert(t(5, 0), 20, 2);
        assert_eq!(old.first().map(|e| e.buf), Some(1));
        assert_eq!(p.len(), 1);
        assert_eq!(p.bytes(), 20);
    }

    #[test]
    fn best_prefers_full_then_the_deepest_carried_checkpoint() {
        let mut p: Pool<u8> = Pool::new(1000);
        // a 100-token turn-1 prompt with checkpoints at 32, 64 and 96 (the last = its message boundary)
        p.insert_marked(t(100, 0), 100, 1, vec![32, 64, 96], None, vec![(100, 0)]);
        // turn 2 shares the first 98 tokens, then differs: resume at the carried checkpoint 96
        let mut turn2 = t(140, 0);
        turn2[98] = 5000;
        assert_eq!(p.best(&turn2, 139), Some(Resume { i: 0, at: 96, full: false }));
        // a shorter prompt sharing 70 tokens: checkpoint 64
        let mut short = t(80, 0);
        short[70] = 7000;
        assert_eq!(p.best(&short, 79).map(|r| r.at), Some(64));
        // the same prompt again (end = its plen-1 = 100): the full entry
        let again = t(101, 0);
        assert_eq!(p.best(&again, 100), Some(Resume { i: 0, at: 100, full: true }));
        // nothing shared below the first checkpoint: no resume
        let mut other = t(50, 0);
        other[10] = 9;
        assert_eq!(p.best(&other, 49), None);
        // end caps a checkpoint resume too
        assert_eq!(p.best(&turn2, 50).map(|r| r.at), Some(32));
    }

    #[test]
    fn identical_operation_sequences_give_identical_state() {
        // the TP contract: every rank applies the same replicated inputs and must end in the same state
        let run = || {
            let mut p: Pool<u32> = Pool::new(100);
            let mut log = Vec::new();
            for k in 0..20u32 {
                let toks = t(1 + (k as usize % 7), k * 3);
                // PFXPOOL3: the entrant's token count joins the victim order; a refusal (None) is replicated too
                if let Some(ev) = p.make_room(17 + (k as u64 % 5), toks.len()) {
                    for e in ev { log.push(e.buf); }
                }
                p.insert(toks, 17 + (k as u64 % 5), k);
                if k % 3 == 0 { p.touch(0); }
            }
            (log, p.bytes(), p.len())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn lineage_clip_and_compare() {
        let a: Segs = vec![(100, 1), (300, 2)];
        assert_eq!(clip(&a, 50), Some(vec![(50, 1)]));
        assert_eq!(clip(&a, 100), Some(vec![(100, 1)]));
        assert_eq!(clip(&a, 200), Some(vec![(100, 1), (200, 2)]));
        assert_eq!(clip(&a, 301), None);
        let b: Segs = vec![(100, 1), (250, 3)];
        assert!(same_rows(&a, &b, 100));
        assert!(!same_rows(&a, &b, 101));
        assert!(same_rows(&a, &b, 0));
    }

    #[test]
    fn common_rows_walks_equal_segments() {
        assert_eq!(common_rows(&[(100, 1), (300, 2)], &[(100, 1), (250, 2)]), 250);
        assert_eq!(common_rows(&[(100, 1), (300, 2)], &[(100, 1), (250, 3)]), 100);
        assert_eq!(common_rows(&[(100, 1)], &[(50, 1), (80, 2)]), 50);
        assert_eq!(common_rows(&[(100, 9)], &[(100, 1)]), 0);
        assert_eq!(common_rows(&[], &[(100, 1)]), 0);
    }

    #[test]
    fn chained_entries_share_rows_and_evict_together() {
        let mut p: Pool<u32> = Pool::new(100_000);
        // nested fills: 4K entry (gen 1), then an 8K prompt that resumed from its rows at 4095 (gen 2 after)
        let (a, _) = p.insert_marked(t(4147, 0), 4147, 10, vec![], None, vec![(4147, 1)]);
        let s8: Segs = vec![(4095, 1), (8243, 2)];
        let (ia, k) = p.best_parent(&s8, 16).unwrap();
        assert_eq!((p.get(ia).unwrap().id, k), (a, 4095));
        let (b, _) = p.insert_marked(t(8243, 0), 8243 - 4032, 20, vec![], Some((a, 4032)), s8.clone());
        let s16: Segs = vec![(4095, 1), (8191, 2), (16435, 3)];
        let (ib, k) = p.best_parent(&s16, 16).unwrap();
        assert_eq!((p.get(ib).unwrap().id, k), (b, 8191));
        let (c, _) = p.insert_marked(t(16435, 0), 16435 - 8128, 30, vec![], Some((b, 8128)), s16);
        let ic = p.index_of(c).unwrap();
        assert_eq!(p.get(ic).unwrap().depth, 2);
        // rows 0..16436 of the 16K entry: own 8128.., then the 8K entry's 4032..8128, then the 4K entry's 0..4032
        let parts: Vec<(u64, usize, usize)> = p.row_parts(ic, 16436).unwrap().into_iter()
            .map(|(i, lo, hi)| (p.get(i).unwrap().id, lo, hi)).collect();
        assert_eq!(parts, vec![(c, 8128, 16436), (b, 4032, 8128), (a, 0, 4032)]);
        // a resume at 3000 rows needs only the root
        let parts: Vec<u64> = p.row_parts(ic, 3000).unwrap().into_iter().map(|(i, _, _)| p.get(i).unwrap().id).collect();
        assert_eq!(parts, vec![a]);
        // the depth cap
        assert!(p.best_parent(&vec![(4095, 1), (8191, 2), (16435, 3), (20000, 4)], 2).map_or(true, |(i, _)| p.get(i).unwrap().depth < 2));
        // touching the deepest entry keeps its ancestors warm: an unrelated newer entry becomes the LRU
        let (d, _) = p.insert_marked(t(10, 500), 10, 40, vec![], None, vec![(10, 7)]);
        p.touch(ic);
        assert_eq!(p.lru(), Some(d));
        assert_eq!(p.pop_lru().iter().map(|e| e.buf).collect::<Vec<_>>(), vec![40]);
        // evicting the root takes the whole chain
        let ia = p.index_of(a).unwrap();
        let mut bufs: Vec<u32> = p.remove_at(ia).iter().map(|e| e.buf).collect();
        bufs.sort();
        assert_eq!(bufs, vec![10, 20, 30]);
        assert_eq!(p.bytes(), 0);
        assert!(p.is_empty());
    }
}
