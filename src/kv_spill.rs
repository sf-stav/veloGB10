//! SSD spill tier for evicted lane snapshots (`--kv-spill`), LANERES2.
//!
//! Problem (owner directive 2026-10-15, after LANERES1): the reservation policy keeps a
//! conversation's prefix cache while it waits, but when conversations outnumber lanes a HARD
//! eviction still happens — the queue head out-waits `--lane-reserve-patience` and takes a
//! reserved lane, or an unreserved victim is chosen. Today the victim's snapshot is dropped and
//! its next request re-prefills from scratch. Andrei's simulator (extended for this session,
//! `contrib/alexei/slot_policy_sim.py --kv-spill-gb`) prices that residual at 15–36% of trace
//! wall time across the agent-load cells even with reservation ON — far above the ~5% gate.
//!
//! Design: when a snapshot is evicted, its bytes are saved to NVMe by a background writer
//! thread (never in the serving round, never on the compute stream); when the conversation's
//! next request arrives and its snapshot is NOT in any free lane, the store is consulted: if
//! `bytes / nvme_read_BW` beats `tokens / prefill_rate` the snapshot is restored (bit-exact
//! through the engine's per-slot checkpoint path), else it is dropped and the re-prefill
//! counter incremented. A miss is never an error: the admission path falls back to today's
//! re-prefill (graceful, no refusals).
//!
//! SPILLDUR (2026-11-22, the 215k-token session in server.log): the capture is QUEUED to the writer thread, so a
//! session whose next request arrived before its ~3.9 GiB capture landed (`pending` = queued-but-not-durable bytes)
//! found its own snapshot in the policy, no file on disk, and ate a silent miss + a 162 s re-prefill (1326 tok/s
//! measured) — a 551-token helper request had taken the lane between the two turns and the turn gap was shorter
//! than the write. Now the store tracks the conv keys still in the queue and a restore that hits one JOINS the
//! write: one bounded condvar wait — the cost model's headroom (`tokens/prefill_rate − bytes/read_BW`, the same
//! numbers `restore_wins` prices), capped at `SPILL_WAIT_MAX_SECS` — then reads. Past the budget it proceeds with
//! the miss, exactly as before. The wait is single-rank only: TP captures go through `put_sync` at the lockstep
//! point, so under TP the queue is always empty and no live-queue state branches across ranks (no new rendezvous).
//!
//! SPILLDUR2 (cost-weighted retention): the cap evicts ASCENDING by snapshot depth (fewest tokens first,
//! LRU within equal tokens) and REFUSES an entrant it could only admit by evicting a DEEPER snapshot — a 551-token
//! helper must not shift out a 215k-token session, whose return then cost the 162 s re-prefill above.
//!
//! SPILLTRUNC (2026-11-22): a snapshot DEEPER than the incoming prompt is a hit, not a miss — the measured case
//! was a harness tail-truncation/regenerate: the session returned 452 tokens shorter than its captured depth and
//! the depth guard (`n > prompt.len()` = miss) ate a ~190 s full re-prefill although rows 0..prompt-depth are
//! byte-identical to what a fresh prefill computes. Captures now carry the lane's WP16 checkpoints (positions +
//! prefix keys + the exact recurrent-state words); when the chain matches at n > m the restore runs at the deepest
//! carried checkpoint q <= m whose prefix key proves prompt[..q] == snapshot[..q], reading ONLY rows 0..q (plus
//! the checkpoint at q — the snapshot's own state/hist sit at depth n and are unusable at q, so a truncated
//! restore is rows + one checkpoint, never the state planes). q comes from replicated inputs only (prompt tokens
//! + policy state), the read is priced at the truncated bytes, and under TP the outcome rides the same
//! SpillFlag agree as every other restore. Without checkpoints in the file (a wp16-off capture, a foreign grid
//! C, or no boundary at-or-below m) the depth guard still misses — with a line saying exactly why.
//!
//! SPILLGC (2026-11-22): the Policy starts EMPTY every boot and the store never looked at its own directory —
//! files from previous boots (or a same-dir sharing instance) were invisible to the cap forever, and every restart
//! added another ~cap of untracked bytes (measured: 182 GiB / 99 files under `--kv-spill-gb 32`, oldest predating
//! several restarts). `Store::boot_scan` + `boot_finish` (called at boot, before serving, from the builder) now
//! self-clean: files NAMED for this rank are header-validated exactly as far as the READ path validates (magic,
//! version, lengths, tokens range — not the payload CRC: that would re-read GiB at boot; a CRC-mismatching file is
//! swept by the existing read-failure path), the VALID ones are ADOPTED into the policy (so a restore right after
//! boot hits them), the malformed ones are DELETED, and if the adopted total exceeds the cap the SPILLDUR2 order
//! sweeps it NOW (ascending depth: the deep sessions are kept, the cheap-to-re-read shallow ones go first) with
//! the victims' files deleted. Same-dir coexistence is ADOPT-THEN-CAP: names claiming another rank are never
//! touched; a same-rank sharer joins the same cap through adoption. Under TP the per-rank adoption can only fork
//! the replicated policy if the ranks' file sets diverge manually — capture-agree keeps them equal by construction
//! (a failed write drops the key everywhere) and the restore-side SpillFlag agrees every read.
//!
//! This module holds the pure policy (cap, cost-weighted retention, cost model) and the disk store façade, imports
//! only std, and is unit-tested here without a GPU — `rustc --edition 2021 --test src/kv_spill.rs`
//! runs standalone, same as `lane_reserve.rs`. The engine hooks live in `exl3_serve.rs`
//! (capture at the take-over site, restore in the admission block); TP ranks each write their
//! own `rank<N>` files, so TP=1/2/4 need no policy change.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// ~0.5 GiB of KV + states per 38K prompt tokens (Flash-Next 27B EXL3 4.05bpw, TP=1 shard;
/// the sim's calibration). A per-token figure read from the model would be better; this is the
/// documented default and `--kv-spill-gb` sizing uses it.
pub const KV_BYTES_PER_TOKEN: u64 = 14_300;

/// .13-measured restore-read bandwidth (O_DIRECT 2-GiB dd, 2026-10-16): ~5.1 GB/s. The cost
/// model is hardware-class, not per-boot; the replay cells re-measure it if it moves.
pub const NVME_READ_BPS: u64 = 5_100_000_000;
/// The admission prefill rate class the cost model compares the read against (Flash-Next 27B
/// EXL3, TP=1, .13, agent trace): tok/s. A restore wins when bytes/BPS < tokens/TPS.
pub const PREFILL_TPS: f64 = 1600.0;

/// SPILLDUR: the hard cap on a restore-side wait for a still-queued capture to land. The motivating failure had a
/// 3.9 GiB capture still queued seconds after the eviction, at the session's next request; a few more seconds of
/// queue drain buys the restore (134 s of prefill on that prompt) — past this a request should not stall on the
/// queue even when the cost model has more headroom left.
pub const SPILL_WAIT_MAX_SECS: f64 = 5.0;

/// File magic: "GB10KSP1" (GB10 KV-spill, format 1).
const MAGIC: &[u8; 8] = b"GB10KSP1";
/// Header: magic(8) + version u32 + tokens u64 + payload_len u64 + crc u64 = 36 bytes.
const HEADER_LEN: u64 = 8 + 4 + 8 + 8 + 8;

/// SPILLTRUNC: the engine payload format the tiers speak — lives HERE (not only in the engine) so the boot
/// scanner validates exactly what the restore accepts; `exl3_forward::tensors` re-exports it. Version 6 appended
/// the checkpoint section after the payload's hist tail:
///
/// ```text
/// payload = inner36(GB10SPIL, ver, tokens, nreg, total) | regions(total) | hist(8)
///          | nck u32 | ck_c u32 | fixed u64 | table: nck x (pos u64, prefix_key u64, hist i32 x2)
///          | nck x ckpt_bytes (the exact ckpt_save words, table order)
/// ```
///
/// `fixed` = the FILE bytes that do not scale with rows (both headers + the state planes + hist + this section):
/// [`Policy::trunc_read`] prices a truncated restore with it. A version-5 file carries no section: full restores
/// of it are dropped by the engine's version check, so the boot scanner sweeps it and frees the disk.
pub const SPILL_PAYLOAD_VERSION: u32 = 6;
/// Bytes per checkpoint table entry: pos u64 + prefix_key u64 + hist i32 x2.
pub const CK_TABLE_ENTRY_BYTES: u64 = 8 + 8 + 8;
/// Bytes of the checkpoint section trailer before the table: nck u32 + ck_c u32 + fixed u64.
pub const CK_TRAILER_BYTES: u64 = 4 + 4 + 8;
/// Host staging used one chunk at a time for the device-to-host copy.
pub(crate) const CAPTURE_CHUNK_BYTES: usize = 64 << 20;
/// Leave room above the watchdog for sampling, page cache writeback and concurrent host allocations.
const CAPTURE_HEADROOM_BYTES: u64 = 512 << 20;

/// Serialized v6 file length, including both headers, PLE history and the checkpoint section.
pub(crate) fn capture_file_bytes(region_bytes: usize, checkpoint_bytes: usize, checkpoints: usize) -> usize {
    HEADER_LEN as usize + 36 + region_bytes + 8 + CK_TRAILER_BYTES as usize
        + checkpoint_bytes + checkpoints * CK_TABLE_ENTRY_BYTES as usize
}

/// Host memory room for a single spill capture above the watchdog floor, guard margin and pinned staging chunk.
pub(crate) fn host_capture_budget(available: u64, watchdog_floor: u64) -> u64 {
    available
        .saturating_sub(watchdog_floor)
        .saturating_sub(CAPTURE_HEADROOM_BYTES)
        .saturating_sub(CAPTURE_CHUNK_BYTES as u64)
}

/// The smaller of async queue room and host room. Pending buffers are already reflected in MemAvailable.
pub(crate) fn capture_byte_budget(pending: u64, max_pending: u64, available: u64, watchdog_floor: u64) -> u64 {
    max_pending.saturating_sub(pending).min(host_capture_budget(available, watchdog_floor))
}

/// FNV-1a 64: payload integrity without pulling in a crate (std-only module).
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The flags: `--kv-spill <auto|off|PATH> --kv-spill-gb N`.
#[derive(Clone, Debug)]
pub struct Cfg {
    /// false = the tier is off: no capture, no restore, counters stay zero (today's behavior).
    pub on: bool,
    /// Spill directory. `auto` resolves to `/tmp/gb10_kv_spill_<port>`; PATH is used verbatim.
    pub dir: PathBuf,
    /// Store cap in bytes (`--kv-spill-gb`, default 64).
    pub cap_bytes: u64,
    /// NVMe read rate used by the restore cost model (bytes/s; boot-measured class, dd anchor).
    pub nvme_read_bps: u64,
    /// Prefill rate used by the restore cost model (tokens/s).
    pub prefill_tps: f64,
}

impl Cfg {
    pub fn off() -> Self {
        Cfg { on: false, dir: PathBuf::from("/tmp/gb10_kv_spill"), cap_bytes: 0, nvme_read_bps: 5_100_000_000, prefill_tps: 1600.0 }
    }
}

/// The LRU byte-cap policy over the store (pure; the files live in [`Store`]).
/// One tracked snapshot: conv key, on-disk bytes, depth, and (SPILLTRUNC) what a truncated restore needs —
/// `fixed` = the file bytes that do NOT scale with rows (both headers + the state planes + hist + the checkpoint
/// section), `ck_c` = the WP16 chunk width the carried checkpoints were taken at (0 = none carried), and `ck` =
/// (position, prefix key = conv_key(tokens[..position])) ascending. Every field is replicated state: computed
/// from the captured tokens (identical on every rank) or the file (adopted at boot).
#[derive(Clone, Debug)]
pub struct Snap {
    pub conv: u64,
    pub bytes: u64,
    pub tokens: u64,
    pub fixed: u64,
    pub ck_c: usize,
    pub ck: std::sync::Arc<[(u64, u64)]>,
}

impl Default for Snap {
    fn default() -> Self {
        Snap { conv: 0, bytes: 0, tokens: 0, fixed: 0, ck_c: 0, ck: std::sync::Arc::new([]) }
    }
}

/// A restore target found by [`Policy::lookup_at`]. Full = the whole snapshot (today's path); Trunc = the
/// SPILLTRUNC case: restore rows 0..q + the carried checkpoint at q out of an n-deep snapshot (q <= prompt depth,
/// `read_bytes` = the priced truncated read). `saved` = the re-prefill tokens the restore replaces either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    Full { conv: u64, n: usize, bytes: u64 },
    Trunc { conv: u64, q: usize, n: usize, read_bytes: u64 },
}

impl Hit {
    pub fn conv(&self) -> u64 {
        match self {
            Hit::Full { conv, .. } | Hit::Trunc { conv, .. } => *conv,
        }
    }
    /// the depth the lane resumes at (the hit's token count for the hit accounting)
    pub fn saved(&self) -> usize {
        match self {
            Hit::Full { n, .. } => *n,
            Hit::Trunc { q, .. } => *q,
        }
    }
    pub fn read_bytes(&self) -> u64 {
        match self {
            Hit::Full { bytes, .. } => *bytes,
            Hit::Trunc { read_bytes, .. } => *read_bytes,
        }
    }
}

#[derive(Debug, Default)]
pub struct Policy {
    cap_bytes: u64,
    bytes: u64,
    /// SPILLTRUNC: full `Snap` records, front = least recently put.
    lru: VecDeque<Snap>,
}

impl Policy {
    pub fn new(cap_bytes: u64) -> Self {
        Policy { cap_bytes, bytes: 0, lru: VecDeque::new() }
    }

    pub fn len(&self) -> usize {
        self.lru.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lru.is_empty()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// LANERES3: the configured byte cap — the TP capture gate reads it as the
    /// deterministic budget check (identical on every rank, unlike the live writer queue).
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }
    pub fn contains(&self, conv: u64) -> bool {
        self.lru.iter().any(|s| s.conv == conv)
    }

    /// Insert/refresh a snapshot of `bytes` at `tokens` (no carried checkpoints); returns the conv keys shifted
    /// out (victims whose files the caller should delete). See [`Policy::put_snap`] for the retention rules.
    pub fn put(&mut self, conv: u64, bytes: u64, tokens: u64) -> Vec<u64> {
        self.put_snap(Snap { conv, bytes, tokens, ..Default::default() })
    }

    /// Insert/refresh one snapshot record; returns the conv keys shifted out (victims whose files the caller
    /// should delete). Re-putting an existing conv refreshes it and replaces its size and checkpoint table.
    /// SPILLDUR2 (cost-weighted retention): the cap NEVER evicts a deeper snapshot to admit a shallower one —
    /// victims come out ascending by depth (fewest tokens first, LRU order within equal tokens), and if all the
    /// snapshots no deeper than the entrant cannot free enough room, the ENTRANT is refused: no victims, nothing
    /// inserted, policy untouched (the entrant is by definition cheaper to re-read than the snapshots it would
    /// displace — the measured case: a 551-token helper evicting a 215k-token session cost 162 s on return).
    /// A snapshot bigger than the whole cap is likewise refused. Deterministic: the policy is replicated state
    /// under TP, so every rank must reach the same victim set / refusal from the same inputs.
    pub fn put_snap(&mut self, s: Snap) -> Vec<u64> {
        let (conv, bytes, tokens) = (s.conv, s.bytes, s.tokens);
        if bytes > self.cap_bytes {
            return Vec::new();
        }
        let refresh = self.lru.iter().position(|e| e.conv == conv).map(|pos| self.lru.remove(pos).unwrap_or(s.clone()));
        if let Some(old) = &refresh {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        if self.bytes.saturating_add(bytes) <= self.cap_bytes {
            self.lru.push_back(s);
            self.bytes += bytes;
            return Vec::new();
        }
        let mut need = self.bytes + bytes - self.cap_bytes;
        // the victim plan runs on the untouched deque: a refusal must be exactly a no-op
        let mut order: Vec<usize> = (0..self.lru.len()).collect();
        order.sort_by_key(|&i| (self.lru[i].tokens, i)); // ascending depth; LRU front first within equal depths
        let mut victims: Vec<usize> = Vec::new();
        for &i in &order {
            if self.lru[i].tokens > tokens {
                break; // ascending: everything past here is deeper than the entrant
            }
            victims.push(i);
            need = need.saturating_sub(self.lru[i].bytes);
            if need == 0 {
                break;
            }
        }
        if need > 0 {
            // refusal: the eligible (shallower-or-equal) snapshots cannot free enough — undo the refresh's removal
            if let Some(e) = refresh {
                self.bytes += e.bytes;
                self.lru.push_back(e);
            }
            return Vec::new();
        }
        let mut vkeys = Vec::with_capacity(victims.len());
        for i in victims.into_iter().rev() {
            if let Some(e) = self.lru.remove(i) {
                self.bytes = self.bytes.saturating_sub(e.bytes);
                vkeys.push(e.conv);
            }
        }
        self.lru.push_back(s);
        self.bytes += bytes;
        vkeys
    }

    /// SPILLDUR2 pre-check: would `put_snap` admit `(conv, bytes, tokens)` (either directly or by evicting
    /// shallower-or-equal entries)? Pure dry-run (no mutation) so callers can skip GiB-scale D2H + NVMe writes
    /// for shallow entrants that `put_snap` would immediately refuse and unlink. Deterministic on replicated
    /// policy state under TP.
    pub fn would_admit(&self, conv: u64, bytes: u64, tokens: u64) -> bool {
        if bytes > self.cap_bytes {
            return false;
        }
        let cur = match self.lru.iter().find(|e| e.conv == conv) {
            Some(old) => self.bytes.saturating_sub(old.bytes),
            None => self.bytes,
        };
        if cur.saturating_add(bytes) <= self.cap_bytes {
            return true;
        }
        let need = cur + bytes - self.cap_bytes;
        let avail: u64 = self.lru.iter()
            .filter(|e| e.conv != conv && e.tokens <= tokens)
            .map(|e| e.bytes)
            .sum();
        avail >= need
    }

    /// SPILLDUR: the bytes the store recorded for `conv` (None when absent) — the restore-side durability wait
    /// prices the queue behind it with the same numbers `restore_wins` uses.
    pub fn snapshot_bytes(&self, conv: u64) -> Option<u64> {
        self.lru.iter().find(|s| s.conv == conv).map(|s| s.bytes)
    }

    /// SPILLTRUNC: the priced read of a truncated restore at `q` out of `s` — everything fixed plus the rows
    /// pro-rated by depth (rows scale linearly with tokens; the truncation reads exactly the row-region prefixes,
    /// so this over-counts only by the hist and the NOT-read carried checkpoints — a conservative gate).
    pub fn trunc_read(s: &Snap, q: usize) -> u64 {
        let rows = s.bytes.saturating_sub(s.fixed);
        s.fixed + rows * (q as u64).min(s.tokens) / s.tokens.max(1)
    }

    /// SPILLDUR: how long a restore may wait for this snapshot to become durable and STILL be cheaper than the
    /// re-prefill it replaces: the cost model's headroom `tokens/prefill_tps − bytes/nvme_read_bps` (positive
    /// whenever `restore_wins` passed), capped at [`SPILL_WAIT_MAX_SECS`]. 0 = do not wait, take the miss.
    pub fn wait_budget_secs(&self, bytes: u64, tokens: u64, nvme_read_bps: u64, prefill_tps: f64) -> f64 {
        if bytes == 0 || nvme_read_bps == 0 || prefill_tps <= 0.0 {
            return 0.0;
        }
        let t_read = bytes as f64 / nvme_read_bps as f64;
        let t_recompute = tokens as f64 / prefill_tps;
        (t_recompute - t_read).clamp(0.0, SPILL_WAIT_MAX_SECS)
    }

    /// SPILLGC: register snapshots ADOPTED from the store directory at boot and cap the store at once; the
    /// registration order is an EXPLICIT sort (tokens, bytes, conv ascending — no readdir/map iteration order
    /// reaches the state), so under TP every rank that saw the same files lands in the same state. A snapshot
    /// bigger than the whole cap cannot exist under it: it is not tracked and returned as a victim (the caller
    /// deletes the file). If the adopted total still exceeds the cap, entries shift out from the LRU front —
    /// which, in this insertion order, IS the SPILLDUR2 order (ascending depth, cheap-to-re-read first; the deep
    /// sessions are kept) — there is no entrant to protect a depth against at boot. Returns the victims
    /// (conv, bytes) INCLUDING the over-cap refusals; the caller deletes their files.
    pub fn boot_register(&mut self, entries: Vec<Snap>) -> Vec<(u64, u64)> {
        let mut es = entries;
        es.sort_by_key(|s| (s.tokens, s.bytes, s.conv));
        let mut victims: Vec<(u64, u64)> =
            es.iter().filter(|s| s.bytes == 0 || s.bytes > self.cap_bytes).map(|s| (s.conv, s.bytes)).collect();
        es.retain(|s| s.bytes > 0 && s.bytes <= self.cap_bytes);
        for s in es {
            self.bytes += s.bytes;
            self.lru.push_back(s);
        }
        while self.bytes > self.cap_bytes && !self.lru.is_empty() {
            if let Some(s) = self.lru.pop_front() {
                self.bytes -= s.bytes;
                victims.push((s.conv, s.bytes));
            }
        }
        victims
    }

    /// The conv's snapshot was restored (or explicitly dropped): take it out of the policy.
    /// Returns true when it was present (a hit).
    pub fn remove(&mut self, conv: u64) -> bool {
        match self.lru.iter().position(|s| s.conv == conv) {
            Some(pos) => {
                if let Some(s) = self.lru.remove(pos) {
                    self.bytes = self.bytes.saturating_sub(s.bytes);
                }
                true
            }
            None => false,
        }
    }

    /// LANERES2: iterate (conv, bytes, tokens), LRU-front order (deterministic).
    pub fn entries(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        self.lru.iter().map(|s| (s.conv, s.bytes, s.tokens))
    }

    /// LANERES2: find the stored snapshot that is a PREFIX of `prompt` and passes the restore
    /// cost model. Returns (conv key, tokens); the caller reads the file, then remove()s the
    /// entry. Pure scan: at most cap / bytes-per-snapshot entries (~hundreds), each one hash
    /// of prompt[..tokens] — a few ms of CPU in the admission path, only on own-snapshot
    /// misses. MRU-first scan: the hottest eviction wins a tie.
    /// SPILLTRUNC: a snapshot deeper than the prompt is no longer an unconditional miss — see [`Policy::lookup_at`]
    /// (with `c = 0` — no chunk grid — truncation is off, so this stays exactly the historical full-hit scan).
    pub fn lookup(&self, prompt: &[u32], nvme_read_bps: u64, prefill_tps: f64) -> Option<(u64, usize)> {
        match self.lookup_at(prompt, 0, nvme_read_bps, prefill_tps) {
            Some(h) => Some((h.conv(), h.saved())),
            None => None,
        }
    }

    /// SPILLTRUNC: the lookup with truncated-restore support. Same MRU-first scan as `lookup`; for an entry
    /// deeper than the prompt (n > m), a hit is offered when the file carries checkpoints taken at THIS build's
    /// chunk width `c` and one of them, at position q <= m, has a prefix key matching conv_key(prompt[..q]) —
    /// the only proof available that the incoming prompt is a true prefix of the captured stream (the policy
    /// never sees the snapshot's tokens, and neither does any other tier; the keys ride the capture, so under TP
    /// every rank computed them from the same tokens). Deepest provable q first; the read is priced at the
    /// truncated bytes against the q tokens it saves. A loss or an unprovable depth continues the scan (an older
    /// shallower snapshot may still be a Full hit). q depends on replicated inputs only (prompt tokens + policy
    /// state), never live queue or file state (ticket rule: TP ranks reach the same Hit without a new agree).
    pub fn lookup_at(&self, prompt: &[u32], c: usize, nvme_read_bps: u64, prefill_tps: f64) -> Option<Hit> {
        let m = prompt.len();
        let mut logged = 0;
        for s in self.lru.iter().rev() {
            let n = s.tokens as usize;
            if n == 0 {
                continue;
            }
            if n > m {
                // SPILLTRUNC: the chain's snapshot is deeper — offer the deepest provable checkpoint at-or-below m
                let conv = s.conv;
                let why = if c == 0 || s.ck_c == 0 {
                    "no checkpoints carried (wp16-off capture or pre-v6 file)"
                } else if s.ck_c != c {
                    "checkpoint grid mismatch"
                } else {
                    ""
                };
                let mut q = 0usize;
                if why.is_empty() {
                    for (pos, key) in s.ck.iter().rev() {
                        let p = *pos as usize;
                        if p == 0 || p > m {
                            continue;
                        }
                        if conv_key(&prompt[..p]) == *key {
                            q = p;
                            break;
                        }
                    }
                }
                if q == 0 {
                    if logged < 2 {
                        logged += 1;
                        crate::drprintln!("[kv-spill] miss: snapshot n={n} vs prompt.len()={m} (deeper by {}, \
                                          truncated restore unavailable: {})", n - m,
                            if why.is_empty() { "no checkpoint at-or-below prompt depth" } else { why });
                    }
                    continue;
                }
                let rb = Policy::trunc_read(s, q);
                if self.restore_wins(rb, q as u64, nvme_read_bps, prefill_tps) {
                    return Some(Hit::Trunc { conv, q, n, read_bytes: rb });
                }
                if logged < 2 {
                    logged += 1;
                    crate::drprintln!("[kv-spill] miss: snapshot n={n} vs prompt.len()={m} (deeper by {}): provable \
                                      truncation at q={q} dropped by the cost model (read {rb} vs saving {q} tokens)",
                                     n - m);
                }
                continue;
            }
            if conv_key(&prompt[..n]) == s.conv {
                if self.restore_wins(s.bytes, s.tokens, nvme_read_bps, prefill_tps) {
                    return Some(Hit::Full { conv: s.conv, n, bytes: s.bytes });
                }
                if logged < 2 {
                    logged += 1;
                    crate::drprintln!("[kv-spill] miss: hash ok, cost model dropped (n={n} bytes={})", s.bytes);
                }
            } else if logged < 2 {
                logged += 1;
                crate::drprintln!("[kv-spill] miss: hash mismatch at n={n} prompt.len()={m}");
            }
        }
        None
    }


    /// The restore cost model: restore only if the NVMe read beats the re-prefill it replaces.
    /// `tokens` is the snapshot's token count, `prefill_tps` the engine's admission prefill rate.
    pub fn restore_wins(&self, bytes: u64, tokens: u64, nvme_read_bps: u64, prefill_tps: f64) -> bool {
        if bytes == 0 {
            return false; // nothing to read
        }
        if nvme_read_bps == 0 || prefill_tps <= 0.0 {
            return false;
        }
        let t_read = bytes as f64 / nvme_read_bps as f64;
        let t_recompute = tokens as f64 / prefill_tps;
        t_read < t_recompute
    }
}

/// The disk store: a background writer thread owns all writes/deletes; the serving thread only
/// enqueues (async, never in the round) and, on restore, does one synchronous read — the same
/// blocking an admission prefill already costs. Payloads are opaque to this module: the engine
/// serializes lane snapshots through its per-slot checkpoint path, so bit-exactness is theirs.
/// SPILLDUR: the store also tracks which conv keys are queued-but-not-durable (`pending_ops`, guarded by the
/// condvar the writer notifies after EVERY job it finishes, success or failure); a restore whose lookup hit is
/// still in the queue can `await_durable` for it with a bounded wait instead of eating a silent miss.
#[derive(Clone)]
pub struct Store {
    dir: PathBuf,
    tx: Option<Sender<Job>>,
    pub written: Arc<AtomicU64>,
    pub failed: Arc<AtomicU64>,
    pub durable: Arc<AtomicU64>,
    /// Bytes queued but not yet durable. Bounded by max_pending: a capture over budget is
    /// refused here (Err) and the caller counts a recompute-miss — the UNBOUNDED queue held
    /// GiB-scale payload Vecs and killed the b5x10 spill-ON cell on the memwatch.
    pub pending: Arc<AtomicU64>,
    pub max_pending: u64,
    /// conv -> queued-not-yet-durable writes of that conv (a re-capture of the same chain queues a second write;
    /// the join waits for ALL of them — the last one holds the newest snapshot of the same key). Paired with the
    /// condvar the writer notifies on completion.
    pending_ops: Arc<(Mutex<HashMap<u64, u32>>, Condvar)>,
    /// Test hook: the writer sleeps this many ms before each write, simulating a slow NVMe / a deep queue.
    /// Production leaves it 0 (one relaxed load per job).
    pub write_delay_ms: Arc<AtomicU64>,
}

enum Job {
    Write { conv: u64, path: PathBuf, data: Vec<u8> },
    Delete { path: PathBuf },
}

/// SPILLGC: the outcome of [`Store::boot_scan`] — the valid snapshots THIS rank's names own (full [`Snap`]
/// records, conv-sorted, SPILLTRUNC checkpoint tables included) and the garbage already deleted (count + bytes).
#[derive(Debug, Default)]
pub struct BootScan {
    pub adopted: Vec<Snap>,
    pub swept: usize,
    pub swept_bytes: u64,
}

/// SPILLGC: what [`Store::boot_finish`] did, for the one-time boot log line.
#[derive(Debug, Default, Clone, Copy)]
pub struct BootReport {
    pub adopted: usize,
    pub adopted_bytes: u64,
    pub swept: usize,
    pub swept_bytes: u64,
    pub evicted: usize,
    pub evicted_bytes: u64,
}

/// Who a directory entry provably belongs to (SPILLGC rule 4: only files claiming THIS rank are touched).
enum Owner {
    /// ours, conv parsed from the name
    Ours(u64),
    /// ours (this rank's name or temp) but the key is unparseable / it is a stale tmp — sweep
    Sweep,
    /// anyone else's (another rank, another naming scheme): never touched
    Foreign,
}

/// Classify one directory name against the `conv<16hex>_rank<N>.ksp` (+ writer temp `.ksp.part`) scheme.
fn owner_of(name: &str, rank: usize) -> Owner {
    let (stem, is_part) = match name.strip_suffix(".ksp.part") {
        Some(s) => (s, true),
        None => match name.strip_suffix(".ksp") { Some(s) => (s, false), None => return Owner::Foreign },
    };
    let core = match stem.strip_prefix("conv") { Some(c) => c, None => return Owner::Foreign };
    let (hex, rk) = match core.rsplit_once("_rank") { Some(x) => x, None => return Owner::Foreign };
    if rk != rank.to_string() {
        return Owner::Foreign; // another rank's (or an unparseable/padded rank): its owner sweeps it, not us
    }
    match u64::from_str_radix(hex, 16) {
        Ok(conv) if hex.len() == 16 && !is_part => Owner::Ours(conv),
        _ => Owner::Sweep, // our rank's debris: a malformed key, or a tmp our writer never finished
    }
}

/// SPILLGC + SPILLTRUNC: the checks a boot-adopted file must pass — the outer header's (magic, version, plen,
/// tokens range — the reader's checks minus the payload CRC: verifying it would re-read GiB per file at boot; a
/// CRC-mismatching file is swept by the EXISTING read-failure path on its first lookup), PLUS the payload checks
/// the restore itself performs: inner GB10SPIL magic, version == SPILL_PAYLOAD_VERSION (a version-5 file can never
/// restore on this build — sweep it and free the disk), inner tokens == outer tokens, and the checkpoint section
/// sized against THIS model's `ck_bytes`: len == 36 + 36 + total + 8 + CK_TRAILER + nck*(CK_TABLE + ck_bytes).
/// The checkpoint table is parsed (positions strictly ascending, 0 < pos < tokens) so an adopted entry can offer
/// SPILLTRUNC restores right after boot (measured case: the harness truncation that lost 190 s happened AFTER a
/// restart — pre-trunc-adoption boots could only ever full-restore). `max_tokens` bounds tokens by THIS model's
/// context (the restore's `tokens <= max_pos` check); 0 = unbounded (tests). Returns the full policy record
/// (conv filled by the caller — it comes from the file NAME).
fn boot_validate(path: &Path, len: u64, max_tokens: u64, ck_bytes: u64) -> Option<Snap> {
    if len < HEADER_LEN + 36 + 8 + CK_TRAILER_BYTES {
        return None;
    }
    let mut f = std::fs::File::open(path).ok()?;
    let mut hdr = [0u8; 36];
    if f.read_exact(&mut hdr).is_err() {
        return None;
    }
    if &hdr[0..8] != MAGIC || u32::from_le_bytes(hdr[8..12].try_into().ok()?) != 1 {
        return None;
    }
    let tokens = u64::from_le_bytes(hdr[12..20].try_into().ok()?);
    let plen = u64::from_le_bytes(hdr[20..28].try_into().ok()?);
    if plen > (1 << 34) || len != HEADER_LEN + plen {
        return None;
    }
    if tokens == 0 || (max_tokens > 0 && tokens > max_tokens) {
        return None;
    }
    // the engine payload header (offset 36): magic + version + tokens/nreg/total
    let mut ih = [0u8; 36];
    if f.read_exact(&mut ih).is_err() {
        return None;
    }
    if &ih[0..8] != b"GB10SPIL" || u32::from_le_bytes(ih[8..12].try_into().ok()?) != SPILL_PAYLOAD_VERSION {
        return None;
    }
    if u64::from_le_bytes(ih[12..20].try_into().ok()?) != tokens {
        return None;
    }
    let nreg = u64::from_le_bytes(ih[20..28].try_into().ok()?);
    let total = u64::from_le_bytes(ih[28..36].try_into().ok()?);
    if nreg == 0 || total > plen {
        return None;
    }
    // the checkpoint section trailer, right after the hist tail
    use std::io::Seek;
    if f.seek(std::io::SeekFrom::Start(HEADER_LEN + 36 + total + 8)).is_err() {
        return None;
    }
    let mut tr = [0u8; CK_TRAILER_BYTES as usize];
    if f.read_exact(&mut tr).is_err() {
        return None;
    }
    let nck = u32::from_le_bytes(tr[0..4].try_into().ok()?) as u64;
    let ck_c = u32::from_le_bytes(tr[4..8].try_into().ok()?) as usize;
    let fixed = u64::from_le_bytes(tr[8..16].try_into().ok()?);
    if nck > (1 << 16)
        || len != HEADER_LEN + 36 + total + 8 + CK_TRAILER_BYTES + nck * (CK_TABLE_ENTRY_BYTES + ck_bytes)
    {
        return None;
    }
    let (mut ck, mut last, mut prev_key) = (Vec::with_capacity(nck as usize), 0u64, 0u64);
    for _ in 0..nck {
        let mut e = [0u8; CK_TABLE_ENTRY_BYTES as usize];
        if f.read_exact(&mut e).is_err() {
            return None;
        }
        let pos = u64::from_le_bytes(e[0..8].try_into().ok()?);
        let key = u64::from_le_bytes(e[8..16].try_into().ok()?);
        if pos == 0 || pos >= tokens || pos <= last || key == prev_key {
            return None; // positions strictly ascending and inside the snapshot
        }
        last = pos;
        prev_key = key;
        ck.push((pos, key));
    }
    if ck_c == 0 && nck > 0 {
        return None; // checkpoints without the grid they were taken at: unusable, treat as garbage
    }
    Some(Snap { conv: 0, bytes: len, tokens, fixed, ck_c, ck: ck.into() })
}

fn path_for(dir: &Path, conv: u64, rank: usize) -> PathBuf {
    dir.join(format!("conv{conv:016x}_rank{rank}.ksp"))
}

impl Store {
    /// Create (or reuse) the spill directory and start the writer thread.
    pub fn spawn(dir: PathBuf) -> io::Result<Store> {
        Self::spawn_with(dir, 4 << 30)
    }

    /// LANERES2: spawn with an explicit pending-bytes budget — the serve side passes
    /// max(cap/4, 4 GiB), so a capture storm DROPS snapshots (counted on /metrics) instead of
    /// piling their GiB-scale payloads into host RAM behind the NVMe writer.
    pub fn spawn_with(dir: PathBuf, max_pending: u64) -> io::Result<Store> {
        std::fs::create_dir_all(&dir)?;
        let pending = Arc::new(AtomicU64::new(0));
        let (tx, rx) = mpsc::channel::<Job>();
        let written = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicU64::new(0));
        let durable = Arc::new(AtomicU64::new(0));
        let waiting: Arc<(Mutex<HashMap<u64, u32>>, Condvar)> =
            Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
        let write_delay_ms = Arc::new(AtomicU64::new(0));
        let p = Arc::clone(&pending);
        let (w, f, d) = (Arc::clone(&written), Arc::clone(&failed), Arc::clone(&durable));
        let (wait_cv, delay) = (Arc::clone(&waiting), Arc::clone(&write_delay_ms));
        thread::Builder::new()
            .name("kv-spill-writer".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Write { conv, path, data } => {
                            let dm = delay.load(Ordering::Relaxed);
                            if dm > 0 {
                                thread::sleep(Duration::from_millis(dm)); // test hook: writer lag
                            }
                            let tmp = path.with_extension("ksp.part");
                            let ok = std::fs::File::create(&tmp).and_then(|mut f| {
                                f.write_all(&data)?;
                                f.sync_all()
                            });
                            match ok.and_then(|_| std::fs::rename(&tmp, &path)) {
                                Ok(()) => {
                                    w.fetch_add(1, Ordering::Relaxed);
                                    d.fetch_add(data.len() as u64, Ordering::Relaxed);
                                }
                                Err(_) => {
                                    let _ = std::fs::remove_file(&tmp);
                                    f.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            p.fetch_sub(data.len() as u64, Ordering::Relaxed);
                            // SPILLDUR: the conv is durable (or its write failed and the file simply never
                            // arrived) either way — retire the pending mark and wake any waiter.
                            {
                                let mut g = wait_cv.0.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(c) = g.get_mut(&conv) {
                                    *c -= 1;
                                    if *c == 0 {
                                        g.remove(&conv);
                                    }
                                }
                            }
                            wait_cv.1.notify_all();
                        }
                        Job::Delete { path } => {
                            let _ = std::fs::remove_file(path);
                        }
                    }
                }
            })?;
        Ok(Store { dir, tx: Some(tx), written, failed, durable, pending, max_pending, pending_ops: waiting, write_delay_ms })
    }

    /// Enqueue the write of one snapshot: `tokens` + `payload` (bit-exact engine bytes).
    /// `total_len` = the on-disk size (header + payload) for the policy's byte accounting.
    pub fn put(&self, conv: u64, rank: usize, tokens: u64, payload: &[u8]) -> io::Result<u64> {
        let total = HEADER_LEN + payload.len() as u64;
        let mut data = Vec::with_capacity(total as usize);
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&tokens.to_le_bytes());
        data.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        data.extend_from_slice(&fnv1a64(payload).to_le_bytes());
        data.extend_from_slice(payload);
        if let Some(tx) = &self.tx {
            self.queue_write(tx, conv, path_for(&self.dir, conv, rank), total, data)?;
        }
        Ok(total)
    }

    /// LANERES2: enqueue a PREBUILT file body (finalize_header layout) — the single-copy path
    /// for captures; the payload Vec is moved, not re-copied.
    pub fn put_prebuilt(&self, conv: u64, rank: usize, data: Vec<u8>) -> io::Result<u64> {
        let total = data.len() as u64;
        if let Some(tx) = &self.tx {
            self.queue_write(tx, conv, path_for(&self.dir, conv, rank), total, data)?;
        }
        Ok(total)
    }

    /// The one queue point for async writes: budget check, send, pending-bytes bump and the SPILLDUR pending-op
    /// mark all under the pending-ops lock — the mark must exist before the writer can finish the job it is
    /// matching (the writer takes the same lock to retire it), and the budget check shares the lock so two
    /// concurrent captures cannot both pass it and jointly overrun `max_pending`.
    fn queue_write(&self, tx: &Sender<Job>, conv: u64, path: PathBuf, total: u64, data: Vec<u8>) -> io::Result<()> {
        let mut g = self.pending_ops.0.lock().unwrap_or_else(|e| e.into_inner());
        if self.pending.load(Ordering::Relaxed) + total > self.max_pending {
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "spill queue over budget"));
        }
        tx.send(Job::Write { conv, path, data })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "spill writer gone"))?;
        self.pending.fetch_add(total, Ordering::Relaxed);
        *g.entry(conv).or_insert(0) += 1;
        Ok(())
    }

    /// SPILLDUR: does `conv` have a capture queued but not yet durable?
    pub fn is_pending(&self, conv: u64) -> bool {
        self.pending_ops.0.lock().unwrap_or_else(|e| e.into_inner()).get(&conv).copied().unwrap_or(0) > 0
    }

    /// SPILLDUR: join a still-queued capture of `conv`: block (condvar — never a poll loop) until its writer job
    /// completes, up to `budget_secs`. True = durable now (it landed, or there was nothing pending — in both cases
    /// a `get` may be attempted); false = budget spent while still queued (proceed with the miss: waiting further
    /// would cost more than the re-prefill the restore replaces). Bounded unconditionally: if the writer died,
    /// the mark is never retired and this returns false at the budget, exactly the pre-fix behavior.
    pub fn await_durable(&self, conv: u64, budget_secs: f64) -> bool {
        let budget = Duration::from_secs_f64(budget_secs.clamp(0.0, 60.0));
        let (m, cv) = &*self.pending_ops;
        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
        let t0 = Instant::now();
        while g.get(&conv).copied().unwrap_or(0) > 0 {
            let left = budget.saturating_sub(t0.elapsed());
            if left.is_zero() {
                return false;
            }
            let (g2, _) = cv.wait_timeout(g, left).unwrap_or_else(|e| e.into_inner());
            g = g2;
        }
        true
    }

    /// SPILLGC: boot-time scan of the store directory, BEFORE serving (the writer thread is alive but its queue
    /// is empty, so the garbage deletes here complete immediately). Deletes this rank's garbage (bad header,
    /// truncated/padded, tokens out of range, stale `.ksp.part`); collects the valid snapshots for
    /// [`Policy::boot_register`]. Files naming another rank are never touched — same-dir coexistence is
    /// adopt-then-cap. An unreadable directory is "nothing adopted", never an error: the cache tier must not
    /// fail a boot.
    pub fn boot_scan(&self, rank: usize, max_tokens: u64, ck_bytes: u64) -> BootScan {
        let mut out = BootScan::default();
        let rd = match std::fs::read_dir(&self.dir) { Ok(r) => r, Err(_) => return out };
        let mut files: Vec<(String, PathBuf, u64)> = Vec::new();
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if md.is_file() {
                    files.push((e.file_name().to_string_lossy().into_owned(), e.path(), md.len()));
                }
            }
        }
        files.sort_by(|a, b| a.0.cmp(&b.0)); // no FS order reaches the (replicated) outcome
        for (name, path, len) in files {
            let sweep = match owner_of(&name, rank) {
                Owner::Foreign => false,
                Owner::Sweep => true,
                Owner::Ours(conv) => match boot_validate(&path, len, max_tokens, ck_bytes) {
                    Some(mut s) => {
                        s.conv = conv;
                        out.adopted.push(s);
                        false
                    }
                    None => true,
                },
            };
            if sweep && std::fs::remove_file(&path).is_ok() {
                out.swept += 1;
                out.swept_bytes += len;
            }
        }
        out.adopted.sort_by_key(|s| s.conv);
        out
    }

    /// SPILLGC: adopt the (possibly TP-pruned) scan into the policy, enforce the cap NOW in SPILLDUR2 order, and
    /// queue the victims' deletes through the writer (the same ordered path as capture deletes). Returns the
    /// numbers for the one-time boot line; `adopted` counts what the policy KEPT, `evicted` the cap victims
    /// (including snapshots larger than the whole cap, which cannot exist under it).
    pub fn boot_finish(&self, pol: &mut Policy, rank: usize, scan: BootScan) -> BootReport {
        let mut rep = BootReport { swept: scan.swept, swept_bytes: scan.swept_bytes, ..Default::default() };
        let (n, total): (usize, u64) = (scan.adopted.len(), scan.adopted.iter().map(|s| s.bytes).sum());
        for (conv, b) in pol.boot_register(scan.adopted) {
            rep.evicted += 1;
            rep.evicted_bytes += b;
            self.remove(conv, rank);
        }
        rep.adopted = n - rep.evicted;
        rep.adopted_bytes = total - rep.evicted_bytes;
        rep
    }

    /// SPILLGC (TP-safe boot sweep): delete every file in `self.dir` claiming `rank` (`Owner::Ours` and
    /// `Owner::Sweep`, never touching `Owner::Foreign`) without adopting into `Policy`. Under TP, adopting
    /// local disk files independently per rank would fork the replicated `Policy` whenever one node's spill
    /// directory (e.g. `/tmp/gb10_kv_spill`) was cleared on reboot while a peer's survived — causing the rank
    /// with a hit to block in `spill_agree` while the peer skipped it. Sweeping on TP boot enforces the cap
    /// across restarts while keeping every rank's `Policy` identically empty.
    pub fn boot_sweep_all(&self, rank: usize) -> BootReport {
        let mut rep = BootReport::default();
        let rd = match std::fs::read_dir(&self.dir) { Ok(r) => r, Err(_) => return rep };
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if !md.is_file() { continue; }
                let name = e.file_name().to_string_lossy().into_owned();
                if matches!(owner_of(&name, rank), Owner::Ours(_) | Owner::Sweep)
                    && std::fs::remove_file(e.path()).is_ok()
                {
                    rep.swept += 1;
                    rep.swept_bytes += md.len();
                }
            }
        }
        rep
    }

    /// LANERES3 (TP): write one snapshot SYNCHRONOUSLY from the calling (scheduler) thread —
    /// the TP capture hook needs the outcome at the lockstep point where it is agreed
    /// across ranks (`TpSync::spill_agree`), not an async completion that lands at an
    /// arbitrary later step. Same file format and accounting as the writer thread's
    /// Job::Write (tmp + rename); no queue, so no `max_pending` pressure — the TP path has
    /// at most one capture in flight at a time by construction (admission is serial).
    pub fn put_sync(&self, conv: u64, rank: usize, data: &[u8]) -> io::Result<u64> {
        let path = path_for(&self.dir, conv, rank);
        let tmp = path.with_extension("ksp.part");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(data)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        self.written.fetch_add(1, Ordering::Relaxed);
        self.durable.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(data.len() as u64)
    }

    /// SPILLTRUNC: read ONE byte span of a snapshot file (offset 0 = the file start; the caller adds the 36 B
    /// outer header to reach payload bytes, so this module stays layout-agnostic). A truncated restore reads the
    /// row-region prefixes and exactly the checkpoint it consumes through spans like these — "at most
    /// bytes_for(d) + fixed regions", never the surplus rows; a file that vanished or is short mid-span is Err,
    /// exactly like `get` (the caller counts the miss and never fails the round).
    pub fn read_at(&self, conv: u64, rank: usize, off: u64, len: usize) -> io::Result<Vec<u8>> {
        use std::io::Seek;
        let mut f = std::fs::File::open(path_for(&self.dir, conv, rank))?;
        f.seek(std::io::SeekFrom::Start(off))?;
        let mut out = vec![0u8; len];
        f.read_exact(&mut out)?;
        Ok(out)
    }

    /// Synchronous read for a restore: tokens + payload; Err on any miss/corruption (the caller
    /// falls back to re-prefill and counts the miss).
    pub fn get(&self, conv: u64, rank: usize) -> io::Result<(u64, Vec<u8>)> {
        let path = path_for(&self.dir, conv, rank);
        let mut f = std::fs::File::open(&path)?;
        let mut hdr = [0u8; 36];
        f.read_exact(&mut hdr)?;
        if &hdr[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let tokens = u64::from_le_bytes(hdr[12..20].try_into().unwrap());
        let plen = u64::from_le_bytes(hdr[20..28].try_into().unwrap()) as usize;
        let crc = u64::from_le_bytes(hdr[28..36].try_into().unwrap());
        if plen > (1 << 34) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "len absurd"));
        }
        let mut payload = vec![0u8; plen];
        f.read_exact(&mut payload)?;
        if fnv1a64(&payload) != crc {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "crc mismatch"));
        }
        Ok((tokens, payload))
    }

    pub fn remove(&self, conv: u64, rank: usize) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Delete { path: path_for(&self.dir, conv, rank) });
        }
    }
}

/// The conversation key: FNV-1a over the snapshot's prompt tokens. The engine owns token
/// identity; this keeps the spill module std-only and the key stable across ranks (every rank
/// hashes the same prompt, so TP ranks agree on the file names).
/// LANERES2: fill the reserved 36-byte header of a capture buffer — the exact layout `put`
/// writes and `get` reads (MAGIC, version 1, tokens, payload_len = buf.len()-36, crc), so the
/// payload is assembled ONCE and shipped with `put_prebuilt` (no GiB-scale re-copy).
pub fn finalize_header(buf: &mut [u8], tokens: u64) {
    let crc = fnv1a64(&buf[36..]);
    let plen = (buf.len() - 36) as u64;
    buf[0..8].copy_from_slice(MAGIC);
    buf[8..12].copy_from_slice(&1u32.to_le_bytes());
    buf[12..20].copy_from_slice(&tokens.to_le_bytes());
    buf[20..28].copy_from_slice(&plen.to_le_bytes());
    buf[28..36].copy_from_slice(&crc.to_le_bytes());
}

pub fn conv_key(tokens: &[u32]) -> u64 {
    let mut buf = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    fnv1a64(&buf)
}

/// SPILLTRUNC: conv_key of `tokens[..p]` for every boundary p in `positions` (ascending) — the prefix-proof
/// table a capture carries so a later, SHORTER prompt can prove it is a true prefix of the captured stream
/// (the policy only holds the full-stream key, which a prefix cannot match). O(sum p) hashing at capture time
/// (a 223k prompt with the full checkpoint window is a few ms); the result is replicated state — every rank
/// computes it from the same tokens.
pub fn prefix_keys(tokens: &[u32], positions: &[u64]) -> Vec<u64> {
    positions.iter().map(|&p| conv_key(&tokens[..(p as usize).min(tokens.len())])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Cost model: admits only when the read is strictly cheaper than the re-prefill.
    #[test]
    fn restore_cost_model() {
        let p = Policy::new(1 << 30);
        // 0.5 GiB at 5.1 GB/s = ~104 ms; 38K tokens at 1600 tps = ~23.8 s -> restore wins.
        assert!(p.restore_wins(1 << 29, 38_000, 5_100_000_000, 1600.0));
        // Tiny NVMe: recompute is cheaper -> drop.
        assert!(!p.restore_wins(1 << 29, 38_000, 10_000_000, 1600.0));
        // Boundary: equal is a drop (strictly cheaper required).
        let bytes = (38_000.0 / 1600.0 * 5.1e9) as u64;
        assert!(!p.restore_wins(bytes, 38_000, 5_100_000_000, 1600.0));
        assert!(p.restore_wins(bytes - 1, 38_000, 5_100_000_000, 1600.0));
        // Degenerate rates never admit.
        assert!(!p.restore_wins(1, 1, 0, 1600.0));
        assert!(!p.restore_wins(1, 1, 5_100_000_000, 0.0));
    }

    // 2. LRU cap: shift-out returns the oldest victim; refresh moves + resizes; bytes exact.
    #[test]
    fn lru_cap_and_refresh() {
        let mut p = Policy::new(100);
        assert!(p.put(1, 40, 1).is_empty());
        assert!(p.put(2, 40, 1).is_empty());
        assert_eq!(p.put(3, 40, 1), vec![1]); // oldest shifted out
        assert_eq!(p.len(), 2);
        assert!(!p.contains(1));
        assert!(p.contains(2) && p.contains(3));
        assert_eq!(p.put(2, 30, 1), Vec::<u64>::new()); // refresh + shrink: no victim
        assert_eq!(p.bytes(), 70);
        assert_eq!(p.put(4, 40, 1), vec![3]); // 2 refreshed, so 3 is now the LRU
        assert!(p.contains(2) && p.contains(4));
        assert!(!p.contains(3));
        assert_eq!(p.put(5, 200, 1), Vec::<u64>::new()); // bigger than the cap: refused
        assert!(!p.contains(5));
        assert_eq!(p.bytes(), 70);
        assert!(p.remove(4));
        assert!(!p.remove(4));
        assert_eq!(p.bytes(), 30);
    }


    // 2b. lookup(): the stored snapshot that prefixes the prompt, cost-model gated.
    #[test]
    fn lookup_prefix_and_cost() {
        let mut p = Policy::new(1 << 20);
        let snap: Vec<u32> = vec![7, 8, 9, 10, 11];
        let key = conv_key(&snap);
        assert!(p.put(key, 4096, snap.len() as u64).is_empty());
        // same conversation grown: the prompt starts with the snapshot
        let mut prompt = snap.clone();
        prompt.extend_from_slice(&[12, 13, 14]);
        // 4096 bytes at 1e10 B/s reads in 0.4 us; 5 tokens at 5 tps = 1 s: restore wins
        assert_eq!(p.lookup(&prompt, 10_000_000_000, 5.0), Some((key, 5)));
        // cost model off (read slower than re-prefill): no restore
        assert_eq!(p.lookup(&prompt, 1, 5.0), None);
        // an unrelated prompt does not match
        assert_eq!(p.lookup(&[1, 2, 3, 4, 5], 10_000_000_000, 5.0), None);
    }

    // 3. Store roundtrip: put -> get is byte-identical, tokens intact; wrong conv misses.
    #[test]
    fn store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let st = Store::spawn(dir.clone()).unwrap();
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let total = st.put(0xdead_beef, 2, 38_000, &payload).unwrap();
        assert_eq!(total as usize, payload.len() + 36);
        // The writer is async: poll briefly for durability (test-only busy wait on a counter).
        for _ in 0..2000 {
            if st.written.load(Ordering::Relaxed) == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(st.written.load(Ordering::Relaxed), 1);
        assert_eq!(st.durable.load(Ordering::Relaxed), payload.len() as u64 + 36);
        let (tokens, back) = st.get(0xdead_beef, 2).unwrap();
        assert_eq!(tokens, 38_000);
        assert_eq!(back, payload);
        assert!(st.get(0xfeed_face, 2).is_err());
        st.remove(0xdead_beef, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 4. Corruption: a truncated/garbled file is Err (graceful recompute), never a panic.
    #[test]
    fn corrupt_file_is_graceful() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = path_for(&dir, 7, 0);
        std::fs::write(&p, b"garbage not a spill file").unwrap();
        let st = Store { dir: dir.clone(), tx: None, written: Arc::new(AtomicU64::new(0)),
                         failed: Arc::new(AtomicU64::new(0)), durable: Arc::new(AtomicU64::new(0)),
                         pending: Arc::new(AtomicU64::new(0)), max_pending: u64::MAX,
                         pending_ops: Arc::new((Mutex::new(HashMap::new()), Condvar::new())),
                         write_delay_ms: Arc::new(AtomicU64::new(0)) };
        assert!(st.get(7, 0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 5. Conv key: stable across calls and order-sensitive (prompt identity).
    #[test]
    fn conv_key_stable() {
        let a = vec![1u32, 2, 3, 0xdead_beef];
        let b = a.clone();
        assert_eq!(conv_key(&a), conv_key(&b));
        assert_ne!(conv_key(&a), conv_key(&[2u32, 1, 3, 0xdead_beef]));
        assert_ne!(conv_key(&a), conv_key(&a[..3]));
    }

    // 6. SPILLDUR2 (cost-weighted cap, spill side): a shallower snapshot never evicts a deeper one; a deeper
    // entrant evicts the SHALLOWEST first (LRU within equal depths); a refusal is exactly a no-op.
    #[test]
    fn cost_weighted_cap_never_evicts_deeper() {
        let mut p = Policy::new(100);
        let (deep, mid, helper) = (0xdead_0001u64, 0xdead_0002u64, 0xdead_0003u64);
        assert!(p.put(deep, 50, 5000).is_empty()); // the session's snapshot
        assert!(p.put(mid, 49, 1000).is_empty());  // 99 of the 100-byte cap used
        // the 551-token helper capture (server.log) would have to evict BOTH (47 bytes over): refused,
        // policy untouched — nothing shifted out, the helper not tracked
        assert_eq!(p.put(helper, 48, 551), Vec::<u64>::new());
        assert!(!p.contains(helper));
        assert!(p.contains(deep) && p.contains(mid));
        assert_eq!(p.bytes(), 99);
        // an equal-depth entrant MAY displace the equal-depth entry (not deeper): mid (1000 tok) goes,
        // never deep (5000 tok)
        assert_eq!(p.put(helper, 48, 1000), vec![mid]);
        assert!(p.contains(deep) && p.contains(helper) && !p.contains(mid));
        // a deeper entrant evicts the shallowest eligible first: helper (551 tok) before deep (5000)
        assert_eq!(p.put(0xdead_0004, 45, 6000), vec![helper]);
        assert!(p.contains(deep) && !p.contains(helper));
    }

    // 7. SPILLDUR: a restore whose lookup hit is still in the writer queue JOINS the write (test hook: the writer
    // sleeps `write_delay_ms` per job — the server.log race with the clock made testable).
    #[test]
    fn lookup_joins_pending_capture() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_join_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let st = Store::spawn(dir.clone()).unwrap();
        st.write_delay_ms.store(250, Ordering::Relaxed);
        let snap: Vec<u32> = (0..64u32).collect();
        let conv = conv_key(&snap);
        let payload: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        let total = st.put(conv, 0, snap.len() as u64, &payload).unwrap();
        // the capture path registers the policy entry the moment the write is QUEUED (exl3_serve pool.rs)
        let mut pol = Policy::new(10_000);
        pol.put(conv, total, snap.len() as u64);
        // the session's next request: the policy finds the snapshot, the file is still queued —
        // without the join this get() miss is the silent 162 s re-prefill
        let mut full = snap.clone();
        full.extend_from_slice(&[64, 65, 66]);
        let (hit, n) = pol.lookup(&full, NVME_READ_BPS, PREFILL_TPS).unwrap();
        assert_eq!((hit, n), (conv, 64));
        assert!(st.is_pending(conv));
        assert!(st.get(conv, 0).is_err());
        // a budget shorter than the writer lag: take the miss, still queued
        assert!(!st.await_durable(conv, 0.02));
        assert!(st.is_pending(conv));
        // the policy-priced budget outlives the lag: the join lands and the read succeeds
        let bytes = pol.snapshot_bytes(conv).unwrap();
        let budget = pol.wait_budget_secs(bytes, n as u64, NVME_READ_BPS, 5.0);
        assert!(budget > 0.25);
        assert!(st.await_durable(conv, budget));
        assert!(!st.is_pending(conv));
        let (tokens, back) = st.get(conv, 0).unwrap();
        assert_eq!(tokens, 64);
        assert_eq!(back, payload);
        // a conv with nothing queued resolves true immediately (budget irrelevant): the read may proceed
        assert!(st.await_durable(0xabcd, 0.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 8. SPILLDUR: the wait budget is the cost-model headroom, capped; never positive when the restore
    // would not have won.
    #[test]
    fn wait_budget_is_the_cost_model_headroom() {
        let p = Policy::new(1 << 20);
        // the motivating 215k prompt: ~134 s of re-prefill vs a ~0.76 s read — the hard cap applies
        let b = p.wait_budget_secs(3_900_000_000, 214_641, 5_100_000_000, 1600.0);
        assert!((b - SPILL_WAIT_MAX_SECS).abs() < 1e-9);
        // a small snapshot: the headroom itself is the bound (551 tokens at 1600 tps ~= 0.344 s)
        let b2 = p.wait_budget_secs(8_000_000, 551, 5_100_000_000, 1600.0);
        assert!(b2 > 0.33 && b2 < 0.35);
        // read no cheaper than re-prefill (restore_wins would fail): no wait
        assert_eq!(p.wait_budget_secs(1 << 29, 1, 5_100_000_000, 1600.0), 0.0);
        // degenerate inputs never wait
        assert_eq!(p.wait_budget_secs(0, 1000, 5_100_000_000, 1600.0), 0.0);
        assert_eq!(p.wait_budget_secs(1000, 1000, 0, 1600.0), 0.0);
        assert_eq!(p.wait_budget_secs(1000, 1000, 5_100_000_000, 0.0), 0.0);
    }

    // 8b. SPILLTRUNC: a snapshot deeper than the prompt is a Trunc hit at the DEEPEST provable checkpoint
    // (prefix-key match at q), and every unprovable shape (no ck at-or-below q, changed prefix, foreign grid,
    // c=0, cost-model loss) stays a miss that keeps scanning.
    #[test]
    fn lookup_at_offers_deepest_provable_truncation() {
        let toks: Vec<u32> = (0..200u32).collect();
        let conv = conv_key(&toks);
        let poss = [80u64, 150];
        let keys = prefix_keys(&toks, &poss);
        assert_eq!(keys[0], conv_key(&toks[..80])); // the table rides exactly what lookup replays
        let s = Snap {
            conv,
            bytes: 1_600_000,
            tokens: 200,
            fixed: 100_000,
            ck_c: 16,
            ck: poss.iter().zip(&keys).map(|(p, k)| (*p, *k)).collect(),
        };
        let (rb150, rb80) = (Policy::trunc_read(&s, 150), Policy::trunc_read(&s, 80));
        assert!(rb80 < rb150 && rb150 < s.bytes); // pricing scales with the rows actually read
        let mut p = Policy::new(4 << 20); // the 1.6 MB snapshot must be admissible
        p.put_snap(s);
        // a re-render of exactly the first 150 tokens: deepest provable q = 150
        assert_eq!(p.lookup_at(&toks[..150], 16, NVME_READ_BPS, PREFILL_TPS),
                   Some(Hit::Trunc { conv, q: 150, n: 200, read_bytes: rb150 }));
        // 160 tokens: no checkpoint at 150 < q <= 160 matches (none exist there) — q stays 150
        assert_eq!(p.lookup_at(&toks[..160], 16, NVME_READ_BPS, PREFILL_TPS),
                   Some(Hit::Trunc { conv, q: 150, n: 200, read_bytes: rb150 }));
        // a change AFTER the deepest used checkpoint keeps q (rows 0..q are untouched by it)
        let mut branch = toks[..149].to_vec();
        branch[100] = 9999;
        assert_eq!(p.lookup_at(&branch, 16, NVME_READ_BPS, PREFILL_TPS),
                   Some(Hit::Trunc { conv, q: 80, n: 200, read_bytes: rb80 }));
        // a change BEFORE 80 breaks every key: unprovable, plain miss
        let mut branched_early = toks[..149].to_vec();
        branched_early[10] = 9999;
        assert_eq!(p.lookup_at(&branched_early, 16, NVME_READ_BPS, PREFILL_TPS), None);
        // shallower than the first checkpoint: unprovable
        assert_eq!(p.lookup_at(&toks[..50], 16, NVME_READ_BPS, PREFILL_TPS), None);
        // grid mismatch (this run chunks at 32, the file carried 16) and c=0 (WP16 off): misses
        assert_eq!(p.lookup_at(&toks[..150], 32, NVME_READ_BPS, PREFILL_TPS), None);
        assert_eq!(p.lookup_at(&toks[..150], 0, NVME_READ_BPS, PREFILL_TPS), None);
        // cost-model loss: a read slower than 1 KB/s can never beat re-prefilling 150 tokens
        assert_eq!(p.lookup_at(&toks[..150], 16, 1000, PREFILL_TPS), None);
        // a prompt that EXTENDS the snapshot keeps the historical Full hit (unchanged path)
        let mut grown = toks.clone();
        grown.push(200);
        assert_eq!(p.lookup_at(&grown, 16, NVME_READ_BPS, PREFILL_TPS),
                   Some(Hit::Full { conv, n: 200, bytes: 1_600_000 }));
        // lookup() (c = 0) still sees only Full hits — truncation is strictly lookup_at/opt-in
        assert_eq!(p.lookup(&toks[..150], NVME_READ_BPS, PREFILL_TPS), None);
    }

    // ---- SPILLGC: boot-time self-cleanup (adopt / sweep / cap-at-boot) ----

    /// The v6 writer's on-disk shape: finalize_header over [outer 36][inner 36: GB10SPIL, version, tokens,
    /// nreg=1, total][total region bytes][hist 8][trailer: nck, ck_c, fixed][table nck*24][blob nck*ck_bytes].
    /// `fixed` = the file bytes that do not scale with rows (SPILLTRUNC pricing), exactly what spill_capture writes.
    fn write_ksp6(dir: &Path, conv: u64, rank: usize, tokens: u64, body: &[u8], cks: &[(u64, u64)],
                  ck_c: u32, ck_bytes: usize) -> PathBuf {
        let nck = cks.len() as u32;
        let fixed: u64 = 36 + 36 + 8 + 16 + (nck as u64) * (CK_TABLE_ENTRY_BYTES + ck_bytes as u64);
        let mut buf: Vec<u8> = vec![0u8; 36];
        buf.extend_from_slice(b"GB10SPIL");
        buf.extend_from_slice(&SPILL_PAYLOAD_VERSION.to_le_bytes());
        buf.extend_from_slice(&tokens.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&(body.len() as u64).to_le_bytes());
        buf.extend_from_slice(body);
        buf.extend_from_slice(&[0u8; 8]); // hist
        buf.extend_from_slice(&nck.to_le_bytes());
        buf.extend_from_slice(&ck_c.to_le_bytes());
        buf.extend_from_slice(&fixed.to_le_bytes());
        for (p, k) in cks {
            buf.extend_from_slice(&p.to_le_bytes());
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&0i32.to_le_bytes());
            buf.extend_from_slice(&0i32.to_le_bytes());
        }
        for _ in 0..nck {
            buf.resize(buf.len() + ck_bytes, 7);
        }
        finalize_header(&mut buf, tokens);
        let p = path_for(dir, conv, rank);
        std::fs::write(&p, &buf).unwrap();
        p
    }
    fn write_ksp(dir: &Path, conv: u64, rank: usize, tokens: u64, body: &[u8]) -> PathBuf {
        write_ksp6(dir, conv, rank, tokens, body, &[], 0, 0)
    }

    // 9. Pre-existing valid files are adopted: the policy hits them (FULL hits AND — SPILLTRUNC — truncated hits
    // right after boot), the bytes read back; other ranks' names and foreign names are neither adopted nor touched.
    #[test]
    fn boot_adopts_preexisting_snapshots() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_gc1_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let snap: Vec<u32> = (0..64).collect();
        let conv = conv_key(&snap);
        let body: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        // checkpoints at 24 and 48 of the captured stream, grid 16, 8-byte fake words
        let cks: Vec<(u64, u64)> = vec![(24, conv_key(&snap[..24])), (48, conv_key(&snap[..48]))];
        write_ksp6(&dir, conv, 0, 64, &body, &cks, 16, 8);
        let peer = write_ksp(&dir, 0xDEAD, 1, 64, &body);
        std::fs::write(dir.join("unrelated.bin"), b"x").unwrap();
        let st = Store::spawn(dir.clone()).unwrap();
        let scan = st.boot_scan(0, 1 << 20, 8);
        assert_eq!(scan.adopted.len(), 1);
        let s = &scan.adopted[0];
        assert_eq!((s.conv, s.tokens, s.ck_c), (conv, 64, 16));
        assert_eq!(&*s.ck, &cks[..]); // the truncated-hit table rode the boot parse intact
        // file = outer 36 + inner 36 + body + hist 8 + trailer 16 + 2*(24 + 8)
        assert_eq!(s.bytes, 36 + 36 + 1000 + 8 + 16 + 64);
        assert_eq!(scan.swept, 0);
        let exp_bytes = s.bytes;
        let mut pol = Policy::new(1 << 20);
        let rep = st.boot_finish(&mut pol, 0, scan);
        assert_eq!((rep.adopted, rep.adopted_bytes, rep.swept, rep.evicted), (1, exp_bytes, 0, 0));
        assert!(pol.contains(conv));
        // the restore lookup right after boot hits the adopted snapshot — and the bytes really read back
        let mut prompt = snap.clone();
        prompt.push(99);
        assert_eq!(pol.lookup(&prompt, NVME_READ_BPS, PREFILL_TPS), Some((conv, 64)));
        let (tokens, back) = st.get(conv, 0).unwrap();
        assert_eq!(tokens, 64);
        assert_eq!(&back[..8], b"GB10SPIL");
        assert_eq!(back.len(), (exp_bytes - 36) as usize);
        // a re-render of the first 48 tokens gets the TRUNCATED hit from the adopted table (SPILLTRUNC)
        assert_eq!(pol.lookup_at(&snap[..48], 16, NVME_READ_BPS, PREFILL_TPS).map(|h| h.saved()), Some(48));
        assert!(peer.exists() && dir.join("unrelated.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 10. Every malformed OURS file is swept at boot (junk under a valid name, truncated, padded, tokens 0,
    // a pre-v6 payload version, unparseable key, stale writer temp); a peer rank's garbage and foreign names
    // survive untouched.
    #[test]
    fn boot_sweeps_garbage_keeps_foreign() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_gc2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pl: Vec<u8> = vec![7u8; 1000];
        // junk contents under a valid name
        std::fs::write(path_for(&dir, 0x11, 0), b"garbage, not a spill file").unwrap();
        // truncated: header declares 1000 payload bytes, only 500 landed
        let mut half = vec![0u8; 36 + 1000];
        half[36..36 + 500].copy_from_slice(&pl[..500]);
        finalize_header(&mut half, 64);
        std::fs::write(path_for(&dir, 0x22, 0), &half[..36 + 500]).unwrap();
        // padded: valid file with trailing bytes past the section end
        let mut pad = std::fs::read(write_ksp(&dir, 0x33, 0, 64, &pl)).unwrap();
        pad.extend_from_slice(b"0123456789");
        std::fs::write(path_for(&dir, 0x33, 0), &pad).unwrap();
        // tokens 0: unrestorable
        write_ksp(&dir, 0x44, 0, 0, &pl);
        // a pre-v6 file: same headers, no checkpoint section (version 5) — unrestorable, swept
        let mut v5 = std::fs::read(write_ksp(&dir, 0x56, 0, 64, &pl)).unwrap();
        v5[44..48].copy_from_slice(&5u32.to_le_bytes());
        std::fs::write(path_for(&dir, 0x56, 0), &v5).unwrap();
        // our rank, unparseable key
        std::fs::write(dir.join("convZZ_rank0.ksp"), b"junk").unwrap();
        // a stale writer temp of our rank (the exact with_extension the writer uses)
        std::fs::write(path_for(&dir, 0x55, 0).with_extension("ksp.part"), b"partial").unwrap();
        // a peer rank's GARBAGE file and a foreign name: never ours to touch
        let peer = path_for(&dir, 0x66, 1);
        std::fs::write(&peer, b"garbage of rank 1").unwrap();
        std::fs::write(dir.join("unrelated.txt"), b"x").unwrap();
        let st = Store::spawn(dir.clone()).unwrap();
        let scan = st.boot_scan(0, 1 << 20, 0);
        assert_eq!((scan.adopted.len(), scan.swept), (0, 7), "every malformed ours-file swept");
        assert!(scan.swept_bytes > 0);
        let rep = st.boot_finish(&mut Policy::new(1 << 20), 0, scan);
        assert_eq!((rep.adopted, rep.evicted), (0, 0));
        assert!(peer.exists() && dir.join("unrelated.txt").exists());
        for n in [0x11u64, 0x22, 0x33, 0x44, 0x55, 0x56] {
            assert!(!path_for(&dir, n, 0).exists() && !path_for(&dir, n, 0).with_extension("ksp.part").exists());
        }
        assert!(!dir.join("convZZ_rank0.ksp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 11. The adopted total over the cap is swept NOW in SPILLDUR2 order: the snapshot larger than the whole
    // cap cannot exist under it (refused, file deleted), then ascending depth — the deep session survives while
    // the cheap-to-re-read shallow ones go; victims' files are deleted through the writer queue.
    #[test]
    fn boot_cap_sweeps_ascending_keeps_deep() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_gc3_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base: Vec<u32> = (0..5000u32).collect();
        let (shallow, mid, deep, huge) =
            (conv_key(&base[..10]), conv_key(&base[..1000]), conv_key(&base[..5000]), conv_key(&base[..60]));
        let p_shallow = write_ksp(&dir, shallow, 0, 10, &[1u8; 100]); // 196 B
        let p_mid = write_ksp(&dir, mid, 0, 1000, &[2u8; 1000]); // 1096 B
        let p_deep = write_ksp(&dir, deep, 0, 5000, &[3u8; 2000]); // 2096 B
        let p_huge = write_ksp(&dir, huge, 0, 60, &[4u8; 5000]); // 5096 B > the whole cap
        let st = Store::spawn(dir.clone()).unwrap();
        let scan = st.boot_scan(0, 1 << 20, 0);
        assert_eq!(scan.adopted.len(), 4);
        let mut pol = Policy::new(3200); // fits mid+deep (3192) but not all three valid ones (3388)
        let rep = st.boot_finish(&mut pol, 0, scan);
        assert_eq!((rep.adopted, rep.adopted_bytes, rep.evicted, rep.evicted_bytes), (2, 3192, 2, 196 + 5096));
        assert!(pol.contains(mid) && pol.contains(deep));
        assert!(!pol.contains(shallow) && !pol.contains(huge));
        // the deep session is still the restore answer right after boot
        let mut prompt: Vec<u32> = (0..5000).collect();
        prompt.push(9999);
        assert_eq!(pol.lookup(&prompt, NVME_READ_BPS, PREFILL_TPS), Some((deep, 5000)));
        // the deletes ride the writer queue — bounded wait, then the victims' files are gone and the kept ones remain
        let t0 = std::time::Instant::now();
        while (p_shallow.exists() || p_huge.exists()) && t0.elapsed().as_secs() < 2 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!p_shallow.exists() && !p_huge.exists(), "cap victims' files deleted");
        assert!(p_mid.exists() && p_deep.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 12. An empty (or unreadable) directory is silent and error-free: the tier must never fail a boot.
    #[test]
    fn boot_empty_dir_is_silent() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_gc0_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let st = Store::spawn(dir.clone()).unwrap();
        let scan = st.boot_scan(0, 1 << 20, 0);
        let rep = st.boot_finish(&mut Policy::new(1 << 20), 0, scan);
        assert_eq!((rep.adopted, rep.adopted_bytes, rep.swept, rep.evicted), (0, 0, 0, 0));
        let ghost = Store { dir: PathBuf::from("/gb10-definitely-not-here/gc0"), tx: None,
            written: Arc::new(AtomicU64::new(0)), failed: Arc::new(AtomicU64::new(0)), durable: Arc::new(AtomicU64::new(0)),
            pending: Arc::new(AtomicU64::new(0)), max_pending: u64::MAX,
            pending_ops: Arc::new((Mutex::new(HashMap::new()), Condvar::new())),
            write_delay_ms: Arc::new(AtomicU64::new(0)) };
        assert_eq!(ghost.boot_scan(0, 1 << 20, 0).adopted.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn would_admit_matches_cost_weighted_put_snap_and_tp_boot_sweep_cleans_only_own_rank() {
        let mut pol = Policy::new(100);
        pol.put(1, 60, 200_000);
        pol.put(2, 30, 500);
        // 50-byte shallow helper (400 tokens) needs 40 bytes; only 10 free, and even evicting entry 2 (500 > 400) isn't allowed
        assert!(!pol.would_admit(3, 50, 400));
        // 40-byte entrant at 500 tokens can evict entry 2 (30 B, 500 tok) to free 40 B total
        assert!(pol.would_admit(3, 40, 500));
        // 80-byte deep entrant (250k tokens) can evict both entry 2 and entry 1
        assert!(pol.would_admit(4, 80, 250_000));
        // over-cap is always refused
        assert!(!pol.would_admit(5, 101, 500_000));

        // TP boot sweep deletes rank 0's own .ksp and .ksp.part files without touching rank 1's or foreign files
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_tp_sweep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("conv0000000000000001_rank0.ksp"), b"old-rank0").unwrap();
        std::fs::write(dir.join("conv0000000000000002_rank0.ksp.part"), b"tmp-rank0").unwrap();
        std::fs::write(dir.join("conv0000000000000001_rank1.ksp"), b"keep-rank1").unwrap();
        std::fs::write(dir.join("unrelated.txt"), b"keep-foreign").unwrap();
        let st = Store::spawn(dir.clone()).unwrap();
        let rep = st.boot_sweep_all(0);
        assert_eq!((rep.adopted, rep.swept, rep.evicted), (0, 2, 0));
        assert!(!dir.join("conv0000000000000001_rank0.ksp").exists());
        assert!(!dir.join("conv0000000000000002_rank0.ksp.part").exists());
        assert!(dir.join("conv0000000000000001_rank1.ksp").exists());
        assert!(dir.join("unrelated.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
#[cfg(test)]
mod spill_lookup_tests {
    use super::*;
#[test]
fn lookup_matches_prefix_of_full_prompt() {
    // The returning request's prompt EXTENDS the snapshot (append-only conversation).
    // lookup must scan the FULL prompt: a snapshot matches iff its tokens are a prefix.
    let snap: Vec<u32> = (0..64u32).collect();
    let mut pol = Policy::new(10_000);
    let conv = conv_key(&snap);
    pol.put(conv, 1000, snap.len() as u64);
    let mut full = snap.clone();
    full.extend_from_slice(&[64, 65, 66]);
    // The evicted-state regime: the caller's prefix-cache hit is 0, so the old
    // `&prompt[..0]` slicing found nothing; the full prompt must find it.
    assert!(pol.lookup(&full, NVME_READ_BPS, PREFILL_TPS).is_some());
    // An unrelated prompt must not match.
    let other: Vec<u32> = (100..164u32).collect();
    assert!(pol.lookup(&other, NVME_READ_BPS, PREFILL_TPS).is_none());
}

#[test]
fn capture_file_bytes_and_budgets_enforce_host_and_queue_headroom() {
    let region_bytes = 1_950_000_000usize;
    let ck_bytes = 119_000_000usize;
    let nck = 12usize;
    let base = capture_file_bytes(region_bytes, 0, 0) as u64;
    let full = capture_file_bytes(region_bytes, nck * ck_bytes, nck) as u64;
    assert_eq!(base, HEADER_LEN + 36 + region_bytes as u64 + 8 + CK_TRAILER_BYTES);
    assert_eq!(full, base + (nck as u64) * (CK_TABLE_ENTRY_BYTES + ck_bytes as u64));

    let floor = 5_000_000_000u64;
    let margin = CAPTURE_HEADROOM_BYTES + CAPTURE_CHUNK_BYTES as u64;
    // Below floor + margin -> 0 budget (no underflow)
    assert_eq!(host_capture_budget(floor, floor), 0);
    assert_eq!(host_capture_budget(floor + margin - 1, floor), 0);
    // 8.0 GB available with 5 GB floor -> ~2.40 GB host budget: base (~1.95 GB) fits, full (~3.38 GB) does not
    let avail = 8_000_000_000u64;
    let hb = host_capture_budget(avail, floor);
    assert_eq!(hb, avail - floor - margin);
    assert!(base <= hb && full > hb);

    // Queue room caps the async budget when tighter than host room
    let max_pending = 8u64 << 30;
    assert_eq!(capture_byte_budget(max_pending - (1 << 30), max_pending, 20_000_000_000, floor), 1 << 30);
    assert_eq!(capture_byte_budget(max_pending + 1, max_pending, 20_000_000_000, floor), 0);
    // Host room caps the async budget when tighter than queue room
    assert_eq!(capture_byte_budget(0, max_pending, avail, floor), hb);
}
}
