//! Engram row cache bookkeeping (leg EIO): which record lives in which slot of the pinned, device-mapped
//! slot array, CLOCK replacement, and per-slot step references. The slot BYTES live in
//! `prefetch::EngramIo` (one [`crate::dsv41::gpu::engram::MappedHost`]); this file is pure bookkeeping,
//! owned by the planner thread (no locks here).
//!
//! * Key = `(head << 40) | record` (record < 2^40; a head file holds ~96 M records).
//! * Index: open addressing, linear probing, Fibonacci hash, backward-shift deletion (no tombstones), load
//!   factor <= 1/2.
//! * A slot is evictable only when no unretired step references it (`refs == 0`). A slot whose read is in
//!   flight is always referenced (by the step or warm request that started it), so eviction can never pull
//!   a slot out from under a reader or the GPU.
//! * CLOCK: a hit sets the slot's bit; the hand clears bits and takes the first unreferenced slot with a
//!   clear bit. Free (never used or forgotten) slots are taken first.
//! * `indexed == false` (`--engram-row-cache-mib 0`): nothing is ever found, every request takes a slot
//!   of its own (every row is read from disk); slots still recycle once their steps retire.

const EMPTY: u64 = u64::MAX;

/// Counters of the cache (planner-side, plain integers; the runtime copies them into the metrics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// allocations that found every slot referenced (the caller waits for retirement)
    pub full: u64,
}

pub struct RowCache {
    n_slots: u32,
    indexed: bool,
    /// slot -> key (EMPTY = free)
    slot_key: Vec<u64>,
    refs: Vec<u32>,
    clock: Vec<u8>,
    hand: u32,
    free: Vec<u32>,
    tab_key: Vec<u64>,
    tab_slot: Vec<u32>,
    mask: usize,
    shift: u32,
    pub stats: CacheStats,
}

#[inline]
pub fn key_of(head: usize, record: u64) -> u64 {
    debug_assert!(record < (1u64 << 40) && head < (1 << 23));
    ((head as u64) << 40) | record
}

impl RowCache {
    pub fn new(n_slots: u32, indexed: bool) -> Self {
        assert!(n_slots > 0 && n_slots < u32::MAX);
        let cap = if indexed { (2 * n_slots as usize).next_power_of_two().max(16) } else { 16 };
        RowCache {
            n_slots,
            indexed,
            slot_key: vec![EMPTY; n_slots as usize],
            refs: vec![0; n_slots as usize],
            clock: vec![0; n_slots as usize],
            hand: 0,
            free: (0..n_slots).rev().collect(),
            tab_key: vec![EMPTY; cap],
            tab_slot: vec![0; cap],
            mask: cap - 1,
            shift: 64 - cap.trailing_zeros(),
            stats: CacheStats::default(),
        }
    }
    pub fn n_slots(&self) -> u32 { self.n_slots }
    /// Host bytes of the slot tables and the index (leg MEM1's ledger).
    pub fn host_bytes(&self) -> u64 {
        (self.slot_key.capacity() * 8 + self.refs.capacity() * 4 + self.clock.capacity() + self.free.capacity() * 4 + self.tab_key.capacity() * 8 + self.tab_slot.capacity() * 4) as u64
    }
    pub fn indexed(&self) -> bool { self.indexed }
    pub fn refs(&self, slot: u32) -> u32 { self.refs[slot as usize] }
    /// Slots currently holding a record (resident or loading).
    pub fn occupied(&self) -> usize { self.n_slots as usize - self.free.len() }

    #[inline]
    fn home(&self, key: u64) -> usize { (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize }

    fn find_pos(&self, key: u64) -> Option<usize> {
        let mut i = self.home(key);
        loop {
            let k = self.tab_key[i];
            if k == key { return Some(i); }
            if k == EMPTY { return None; }
            i = (i + 1) & self.mask;
        }
    }
    fn tab_insert(&mut self, key: u64, slot: u32) {
        let mut i = self.home(key);
        while self.tab_key[i] != EMPTY {
            debug_assert_ne!(self.tab_key[i], key);
            i = (i + 1) & self.mask;
        }
        self.tab_key[i] = key;
        self.tab_slot[i] = slot;
    }
    fn tab_remove(&mut self, key: u64) {
        let Some(mut i) = self.find_pos(key) else { return };
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            let k = self.tab_key[j];
            if k == EMPTY { break; }
            let h = self.home(k);
            // move entry j into the hole at i unless its home lies cyclically in (i, j]
            let stays = if i <= j { i < h && h <= j } else { i < h || h <= j };
            if !stays {
                self.tab_key[i] = k;
                self.tab_slot[i] = self.tab_slot[j];
                i = j;
            }
        }
        self.tab_key[i] = EMPTY;
    }

    /// The slot holding `key`, marking it recently used. Counts a hit or a miss.
    pub fn lookup(&mut self, key: u64) -> Option<u32> {
        if self.indexed {
            if let Some(p) = self.find_pos(key) {
                let s = self.tab_slot[p];
                self.clock[s as usize] = 1;
                self.stats.hits += 1;
                return Some(s);
            }
        }
        self.stats.misses += 1;
        None
    }

    /// A slot for `key` (which the caller just missed): a free slot, else the CLOCK victim. The slot is
    /// entered into the index. `None` when every slot is referenced by an unretired step.
    pub fn alloc(&mut self, key: u64) -> Option<u32> {
        let s = match self.free.pop() {
            Some(s) => s,
            None => {
                let n = self.n_slots;
                let mut found = None;
                // two sweeps: the first may only clear bits
                for _ in 0..2 * n {
                    let s = self.hand;
                    self.hand = if self.hand + 1 == n { 0 } else { self.hand + 1 };
                    if self.refs[s as usize] != 0 { continue; }
                    if self.clock[s as usize] != 0 { self.clock[s as usize] = 0; continue; }
                    found = Some(s);
                    break;
                }
                let Some(s) = found else { self.stats.full += 1; return None };
                let old = self.slot_key[s as usize];
                if old != EMPTY && self.indexed { self.tab_remove(old); }
                self.stats.evictions += 1;
                s
            }
        };
        self.slot_key[s as usize] = key;
        self.clock[s as usize] = 1;
        if self.indexed { self.tab_insert(key, s); }
        Some(s)
    }

    #[inline]
    pub fn add_ref(&mut self, slot: u32) { self.refs[slot as usize] += 1; }
    #[inline]
    pub fn release(&mut self, slot: u32) {
        let r = &mut self.refs[slot as usize];
        debug_assert!(*r > 0, "engram cache: release of an unreferenced slot {slot}");
        *r = r.saturating_sub(1);
        if *r == 0 && self.slot_key[slot as usize] == EMPTY { self.free.push(slot); }
    }
    /// Drop the record of `slot` from the index (a failed read: the next request reads it again). The slot
    /// returns to the free list once its last reference is released.
    pub fn forget(&mut self, slot: u32) {
        let k = self.slot_key[slot as usize];
        if k == EMPTY { return; }
        if self.indexed { self.tab_remove(k); }
        self.slot_key[slot as usize] = EMPTY;
        self.clock[slot as usize] = 0;
        if self.refs[slot as usize] == 0 { self.free.push(slot); }
    }
    /// The key a slot holds (None = free / forgotten).
    pub fn key_of_slot(&self, slot: u32) -> Option<u64> {
        let k = self.slot_key[slot as usize];
        (k != EMPTY).then_some(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn hits_misses_and_clock() {
        let mut c = RowCache::new(4, true);
        for k in 0..4u64 {
            assert_eq!(c.lookup(key_of(1, k)), None);
            c.alloc(key_of(1, k)).unwrap();
        }
        assert_eq!(c.stats.misses, 4);
        let s2 = c.lookup(key_of(1, 2)).unwrap();
        assert_eq!(c.key_of_slot(s2), Some(key_of(1, 2)));
        // every slot referenced: alloc fails
        for s in 0..4 { c.add_ref(s); }
        assert_eq!(c.alloc(key_of(2, 9)), None);
        assert_eq!(c.stats.full, 1);
        for s in 0..4 { c.release(s); }
        // all bits set by alloc/lookup: the second sweep evicts something, never a referenced slot
        c.add_ref(s2);
        let v = c.alloc(key_of(2, 9)).unwrap();
        assert_ne!(v, s2);
        assert_eq!(c.lookup(key_of(1, 2)), Some(s2));
        assert_eq!(c.lookup(key_of(2, 9)), Some(v));
        assert_eq!(c.stats.evictions, 1);
        assert_eq!(c.occupied(), 4);
    }

    #[test]
    fn unindexed_never_hits_and_recycles() {
        let mut c = RowCache::new(2, false);
        let a = c.alloc(5).unwrap();
        c.add_ref(a);
        assert_eq!(c.lookup(5), None);
        let b = c.alloc(5).unwrap();
        c.add_ref(b);
        assert_ne!(a, b);
        assert_eq!(c.alloc(6), None);
        c.release(a);
        assert_eq!(c.alloc(6), Some(a));
    }

    #[test]
    fn forget_frees_after_release() {
        let mut c = RowCache::new(2, true);
        let a = c.alloc(7).unwrap();
        c.add_ref(a);
        c.forget(a);
        assert_eq!(c.lookup(7), None);
        assert_eq!(c.occupied(), 1);
        c.release(a);
        assert_eq!(c.occupied(), 0);
    }

    /// The index against a HashMap model under random inserts / evictions / forgets.
    #[test]
    fn index_matches_a_model_under_churn() {
        let mut c = RowCache::new(97, true);
        let mut model: HashMap<u64, u32> = HashMap::new();
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for _ in 0..200_000 {
            let k = key_of((next() % 8) as usize, next() % 300);
            match c.lookup(k) {
                Some(slot) => assert_eq!(model.get(&k), Some(&slot), "hit must agree with the model"),
                None => {
                    assert!(!model.contains_key(&k), "miss on a resident key");
                    if let Some(slot) = c.alloc(k) {
                        model.retain(|_, v| *v != slot);
                        model.insert(k, slot);
                    }
                }
            }
            if next() % 50 == 0 {
                if let Some((&k, &slot)) = model.iter().next() {
                    c.forget(slot);
                    model.remove(&k);
                }
            }
        }
        for (k, slot) in &model { assert_eq!(c.find_pos(*k).map(|p| c.tab_slot[p]), Some(*slot)); }
        let resident = (0..c.n_slots).filter(|&s| c.key_of_slot(s).is_some()).count();
        assert_eq!(resident, model.len());
    }
}
