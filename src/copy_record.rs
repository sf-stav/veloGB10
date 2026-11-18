//! COPYREC1 (2026-11-12): the cross-request copy record (`--copy-span-record on`), host side only.
//!
//! The copy arm (`src/copy_span.rs`) proposes tokens copied from a request's OWN context. This module keeps a
//! server-wide record of the token ids the server read (prompts) and wrote (replies) in EARLIER requests, so a lone
//! request whose own context has nothing to copy can draft from it: a regenerated or repeated request, or an agent
//! re-emitting code / tool calls it wrote in an earlier request, is then served by copy rounds. The target model
//! verifies every copied token exactly as it verifies a context copy (the copy round's chain verify, or the
//! `--copy-span-prefill` stretch), so the record can cost speed, never bytes.
//!
//! * **Ids.** Token ids only, in segments ended by `MARK` (no key spans a mark, no copy crosses one), held in chunks
//!   of `CHUNK_IDS` ids. Every position whose last `KEY_LEN` ids form a key is linked into a fixed-size bucket table
//!   (newest first, `back` = distance to the previous position of the same bucket). A lookup walks at most
//!   `MAX_STEPS` candidates newest first and extends each match backwards up to `MAX_MATCH` ids: the longest match
//!   wins, the newest among equals.
//! * **What goes in.** Per request: its prompt at admission, then `OVERLAP` prompt ids + its reply when the lane is
//!   released (stop, length, cancel, disconnect: whatever was emitted). Runs of `DEDUP_RUN` or more ids the record
//!   already holds are not added again (checked every `DEDUP_STRIDE` ids), so a request sent twice adds nothing
//!   new; each fresh piece carries `OVERLAP` ids on both sides so matches across its joins are found.
//! * **Size.** At most `cap_chunks` chunks (from `--copy-span-record-mib`); past it the OLDEST chunk goes.
//! * **Landing rates.** Server-wide, per match-length band (16-31, 32-63, 64+ ids, `BANDS`): integer fixed-point
//!   moving sums of copied ids that landed and of copies that missed. A copy is offered only when its band's
//!   expected tokens per round beat what the request's own neural rounds deliver lately; otherwise only a probe
//!   (every `probe_every`-th priced-out ask), whose interval doubles (to 1024) while probes miss and resets to 32
//!   when one lands half its ids. A request whose record copy landed nothing leaves the record alone for
//!   `QUIET_ROUNDS` rounds. All gating state is integer and changes only on agreed round outcomes, in program
//!   order: two instances fed the same stream make the same offers (the TP requirement: every rank holds its own
//!   identical record and makes the same proposal every round; the copied ids are still agreed by the existing
//!   pre-verify before a round launches).
//! * **Disk** (`--copy-span-record-dir`). Chunk files `<seq:016x>.ids` (a 32-byte header: magic, version, vocabulary
//!   size, the tokenizer digest's first 16 bytes; then the ids, marks included, little-endian u32) in a directory
//!   made 0700, files 0600, plus `identity` (the tokenizer digest and vocabulary size) and `lock` (held by the
//!   server that writes the record; a second server reads it as it was and writes nothing). A record written under
//!   another tokenizer is never read: the server runs with an empty in-memory record and writes nothing there (one
//!   line, counted). Disk trouble never changes what is held in memory (only persistence stops, counted), so TP
//!   ranks stay identical; at boot the ranks compare the digest of what they loaded and all start empty (memory
//!   only) on a difference.
//!
//! No clock, no randomness, no floating point in any decision. Host cost: a lookup is at most 64 bucket steps with
//! a 64-id extension each; an add is one bucket insert per id plus one dedup lookup per 16 ids.

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// Ends a segment.
pub const MARK: u32 = u32::MAX;
/// Ids one key covers: the shortest match the record can find.
pub const KEY_LEN: usize = 8;
/// A match is extended back at most this far (the top band starts below it).
pub const MAX_MATCH: usize = 64;
/// Bucket candidates a lookup checks at most, newest first.
pub const MAX_STEPS: usize = 64;
/// Ids per chunk (one chunk file's).
pub const CHUNK_IDS: usize = 1 << 20;
/// Ids written before and after each fresh piece (keys across its joins are found).
pub const OVERLAP: usize = MAX_MATCH;
/// A held run at least this long is not added again.
pub const DEDUP_RUN: usize = 2 * OVERLAP;
/// Ids between the starts the dedup checks (runs of DEDUP_RUN ids or more are never missed).
pub const DEDUP_STRIDE: usize = 16;
/// Match lengths that open each band.
pub const BANDS: [usize; 3] = [16, 32, 64];
/// The most ids one record copy takes (the request's copy source; a longer copy searches again at its end).
pub const SNAP_MAX: usize = 8192;
/// Rounds a request leaves the record alone after a record copy that landed nothing.
pub const QUIET_ROUNDS: u64 = 4;
/// `--copy-span-record-mib` default: 64 MiB of ids on disk (about 16 million ids; about twice that in memory).
pub const DEFAULT_MIB: u64 = 64;
/// Chunk file header bytes.
pub const HEAD_BYTES: usize = 32;
/// A full chunk file's bytes (what the cap counts).
pub const CHUNK_FILE_BYTES: u64 = HEAD_BYTES as u64 + 4 * CHUNK_IDS as u64;
const MAGIC: &[u8; 8] = b"VGCPYREC";
const VERSION: u32 = 1;
const PROBE_FIRST: u32 = 32;
const PROBE_MOST: u32 = 1024;
/// Fixed point of the landing sums (x256) and of rates / expected tokens (x65536).
const FX: i64 = 256;
const Q16: i64 = 65536;

// ---------------------------------------------------------------------------------------------
// The ids in memory
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Chunk {
    ids: Vec<u32>,
    /// each position's distance back to the previous position of its key's bucket (0: none)
    back: Vec<u32>,
}

/// A match: its length and the position of the id that follows it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Match {
    pub len: usize,
    pub at: u64,
}

/// The record's ids: segments ended by MARK, keyed by their KEY_LEN-id windows.
pub struct Ids {
    chunks: VecDeque<Chunk>,
    /// the oldest held chunk's number (positions start at first * CHUNK_IDS)
    first: u64,
    /// one past the newest position
    end: u64,
    /// bucket -> its newest position + 1 (0: none)
    heads: Vec<u64>,
    last: [u32; KEY_LEN],
    /// ids since the last mark
    run: usize,
}

#[inline]
fn mix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

impl Ids {
    /// An empty record whose bucket table has 2^bits buckets (fixed: growth never rebuilds it).
    pub fn new(bits: u32) -> Self {
        Ids { chunks: VecDeque::new(), first: 0, end: 0, heads: vec![0; 1usize << bits], last: [MARK; KEY_LEN], run: 0 }
    }

    pub fn base(&self) -> u64 {
        self.first * CHUNK_IDS as u64
    }
    pub fn end(&self) -> u64 {
        self.end
    }
    /// Ids held, marks included.
    pub fn held(&self) -> u64 {
        self.end - self.base()
    }
    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }
    /// Memory the record holds: ids + links (as allocated) + the bucket table.
    pub fn ram_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| (c.ids.capacity() + c.back.capacity()) as u64 * 4).sum::<u64>() + self.heads.len() as u64 * 8
    }

    #[inline]
    pub fn id(&self, p: u64) -> u32 {
        self.chunks[(p / CHUNK_IDS as u64 - self.first) as usize].ids[(p % CHUNK_IDS as u64) as usize]
    }
    #[inline]
    fn back_of(&self, p: u64) -> u32 {
        self.chunks[(p / CHUNK_IDS as u64 - self.first) as usize].back[(p % CHUNK_IDS as u64) as usize]
    }

    #[inline]
    fn bucket(&self, key: &[u32]) -> usize {
        let mut h = 0x5EC0_C0DE_1234_5678u64;
        for &t in key {
            h = (h.rotate_left(29) ^ t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        (mix(h) as usize) & (self.heads.len() - 1)
    }

    /// One id at the end (a full newest chunk takes a new one first).
    pub fn push(&mut self, t: u32) {
        let p = self.end;
        if (p - self.base()) as usize == self.chunks.len() * CHUNK_IDS {
            self.chunks.push_back(Chunk::default());
        }
        self.end = p + 1;
        let mut link = 0u32;
        let keyed = if t == MARK {
            self.run = 0;
            false
        } else {
            self.last.copy_within(1.., 0);
            self.last[KEY_LEN - 1] = t;
            self.run += 1;
            self.run >= KEY_LEN
        };
        if keyed {
            let b = self.bucket(&self.last);
            let h = self.heads[b];
            if h != 0 {
                let d = p - (h - 1);
                if d <= u32::MAX as u64 { link = d as u32; }
            }
            self.heads[b] = p + 1;
        }
        let c = self.chunks.back_mut().unwrap();
        c.ids.push(t);
        c.back.push(link);
    }

    /// `ids` then a mark.
    pub fn add(&mut self, ids: &[u32]) {
        for &t in ids {
            self.push(t);
        }
        self.push(MARK);
    }

    /// The oldest chunk freed (never the newest).
    pub fn drop_oldest(&mut self) -> bool {
        if self.chunks.len() < 2 { return false; }
        self.chunks.pop_front();
        self.first += 1;
        true
    }

    /// Every id gone, the bucket table zeroed; positions start at 0 again.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.heads.iter_mut().for_each(|h| *h = 0);
        self.first = 0;
        self.end = 0;
        self.run = 0;
        self.last = [MARK; KEY_LEN];
    }

    fn key_at(&self, p: u64, key: &[u32]) -> bool {
        let from = p + 1 - KEY_LEN as u64;
        key.iter().enumerate().all(|(i, &t)| self.id(from + i as u64) == t)
    }

    /// Positions whose key may be `key`, newest first, at most MAX_STEPS, each fully held.
    fn walk(&self, key: &[u32]) -> Walk<'_> {
        Walk { r: self, head: self.heads[self.bucket(key)], steps: 0 }
    }

    /// How far back from the key ending at `p` the record matches the tail of `ctx` (KEY_LEN..=MAX_MATCH).
    fn extend(&self, p: u64, ctx: &[u32]) -> usize {
        let mut n = KEY_LEN;
        while n < MAX_MATCH && n < ctx.len() && p >= self.base() + n as u64 {
            let t = self.id(p - n as u64);
            if t == MARK || t != ctx[ctx.len() - 1 - n] { break; }
            n += 1;
        }
        n
    }

    /// The longest match of the tail of `ctx` that has an id after it (the newest of equal length).
    pub fn longest(&self, ctx: &[u32]) -> Match {
        if ctx.len() < KEY_LEN { return Match::default(); }
        let key = &ctx[ctx.len() - KEY_LEN..];
        let mut best = Match::default();
        let mut w = self.walk(key);
        while let Some(p) = w.next() {
            if !self.key_at(p, key) || p + 1 >= self.end || self.id(p + 1) == MARK { continue; }
            let n = self.extend(p, ctx);
            if n <= best.len { continue; }
            best = Match { len: n, at: p + 1 };
            if n == MAX_MATCH { break; }
        }
        best
    }

    /// The ids from `at` up to a mark, the end or `max` ids.
    pub fn copy(&self, at: u64, max: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut q = at;
        while out.len() < max && q < self.end && q >= self.base() {
            let t = self.id(q);
            if t == MARK { break; }
            out.push(t);
            q += 1;
        }
        out
    }

    /// The longest prefix of `s` the record holds in one run: (length, its first position); length 0 = none.
    pub fn held_run(&self, s: &[u32]) -> (usize, u64) {
        if s.len() < KEY_LEN { return (0, 0); }
        let key = &s[..KEY_LEN];
        let mut best = (0usize, 0u64);
        let mut w = self.walk(key);
        while let Some(p) = w.next() {
            if !self.key_at(p, key) { continue; }
            let mut n = KEY_LEN;
            while n < s.len() {
                let q = p + 1 + (n - KEY_LEN) as u64;
                if q >= self.end { break; }
                let t = self.id(q);
                if t == MARK || t != s[n] { break; }
                n += 1;
            }
            if n <= best.0 { continue; }
            best = (n, p + 1 - KEY_LEN as u64);
            if n == s.len() { break; }
        }
        best
    }

    /// Digest of what is held (positions + ids): equal on two ranks iff they hold the same record.
    pub fn digest(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |x: u64| {
            for b in x.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        eat(self.base());
        eat(self.end);
        eat(self.heads.len() as u64);
        for c in &self.chunks {
            for &t in &c.ids {
                eat(t as u64);
            }
        }
        h
    }
}

struct Walk<'a> {
    r: &'a Ids,
    head: u64,
    steps: usize,
}

impl Walk<'_> {
    fn next(&mut self) -> Option<u64> {
        if self.steps == MAX_STEPS || self.head < self.r.base() + KEY_LEN as u64 { return None; }
        self.steps += 1;
        let p = self.head - 1;
        let d = self.r.back_of(p) as u64;
        self.head = if d == 0 { 0 } else { p - d + 1 };
        Some(p)
    }
}

// ---------------------------------------------------------------------------------------------
// Landing rates (integer, server-wide) and the per-request reader state
// ---------------------------------------------------------------------------------------------

/// Copies' landing per match-length band. Integer fixed point; every update comes from agreed round outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rates {
    /// copied ids that landed / copies that missed, moving sums x256 (decay 243/256 per copy round)
    pub landed: [i64; 3],
    pub missed: [i64; 3],
    /// priced-out asks between probes, and asks since the last probe
    pub probe_every: u32,
    pub since: u32,
}

impl Default for Rates {
    /// Priors: about 0.6, 0.8 and 0.9 of a copied id lands, worth half a miss each.
    fn default() -> Self {
        Rates { landed: [192, 512, 1152], missed: [128; 3], probe_every: PROBE_FIRST, since: 0 }
    }
}

/// The band of a match of `len` ids.
pub fn band(len: usize) -> usize {
    let mut b = 0;
    while b + 1 < BANDS.len() && len >= BANDS[b + 1] { b += 1; }
    b
}

/// Tokens a round with a copy of `n` ids is expected to emit at `q` (x65536) a copied id, its own token included
/// (x65536).
pub fn expected_q16(q: i64, n: usize) -> i64 {
    let (mut total, mut reach) = (Q16, Q16);
    for _ in 0..n {
        reach = reach * q >> 16;
        total += reach;
    }
    total
}

impl Rates {
    /// A band's chance (x65536) that a copied id lands when the ids before it did.
    pub fn rate_q16(&self, b: usize) -> i64 {
        let d = self.landed[b] + self.missed[b];
        if d <= 0 { 0 } else { self.landed[b] * Q16 / d }
    }
    /// One copy round of band `b` offered `proposed` ids and `accepted` of them landed.
    pub fn observe(&mut self, b: usize, proposed: usize, accepted: usize, probing: bool) {
        self.landed[b] = self.landed[b] * 243 / 256 + accepted as i64 * FX;
        self.missed[b] = self.missed[b] * 243 / 256 + (accepted < proposed) as i64 * FX;
        if probing {
            self.probe_every = if 2 * accepted >= proposed { PROBE_FIRST } else { (2 * self.probe_every).min(PROBE_MOST) };
        }
    }
    /// Some band's copy of `n` ids is expected to emit more than `head_q16` tokens a round.
    pub fn pays(&self, n: usize, head_q16: i64) -> bool {
        (0..BANDS.len()).any(|b| expected_q16(self.rate_q16(b), n) > head_q16)
    }
    /// Is this priced-out ask the next probe?
    pub fn probe(&mut self) -> bool {
        self.since += 1;
        if self.since < self.probe_every { return false; }
        self.since = 0;
        true
    }
}

/// One record copy round's outcome, applied to the server-wide rates at the lane's next ask (or its release).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Obs {
    pub band: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub probing: bool,
}

/// A request's reads of the record (kept in its copy proposer).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reader {
    /// rounds this request asked its proposer for a draft (with the record armed)
    pub rounds: u64,
    pub quiet_until: u64,
    pub pending: Vec<Obs>,
}

/// What the record offers a request: the copy source (up to SNAP_MAX ids), the match it came from, its band and
/// whether it is a probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offer {
    pub snap: Vec<u32>,
    pub match_len: usize,
    pub band: usize,
    pub probing: bool,
}

// ---------------------------------------------------------------------------------------------
// The record: ids + rates + persistence + counters
// ---------------------------------------------------------------------------------------------

/// What a record's ids mean: the tokenizer's digest (sha256 of tokenizer.json) and the vocabulary size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub digest: [u8; 32],
    pub vocab: u32,
}

impl Identity {
    pub fn from_tokenizer_bytes(bytes: &[u8], vocab: u32) -> Self {
        use sha2::Digest;
        let d = sha2::Sha256::digest(bytes);
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&d);
        Identity { digest, vocab }
    }
    pub fn short(&self) -> String {
        self.digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }
    fn text(&self) -> String {
        let hex: String = self.digest.iter().map(|b| format!("{b:02x}")).collect();
        format!("velogb10 copy record {VERSION}\ndigest {hex}\nvocab {}\n", self.vocab)
    }
}

/// Why the record runs with less than asked (counted, `velogb10_copy_span_record_fallback_total{reason}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// asked without an armed copy arm: the record is off for the boot
    NoCopySpan,
    /// the directory was written under another tokenizer: empty in memory, nothing written there
    Identity,
    /// the directory cannot be made / read / its identity written: memory only
    Dir,
    /// another server holds the directory's lock: read as it was, nothing written
    Locked,
    /// a chunk file was unreadable or malformed: what loaded before it is kept, the rest is cut
    Load,
    /// a write failed: persistence stops for the boot (memory unchanged)
    DiskWrite,
    /// TP: the ranks loaded different records: every rank starts empty, memory only
    TpDigest,
    /// DELETE /v1/copy-record could not remove a chunk file
    Delete,
}

impl Fallback {
    pub const ALL: [Fallback; 8] = [Fallback::NoCopySpan, Fallback::Identity, Fallback::Dir, Fallback::Locked,
                                    Fallback::Load, Fallback::DiskWrite, Fallback::TpDigest, Fallback::Delete];
    pub fn reason(self) -> &'static str {
        match self {
            Fallback::NoCopySpan => "copy_span_off",
            Fallback::Identity => "tokenizer_mismatch",
            Fallback::Dir => "dir_error",
            Fallback::Locked => "locked",
            Fallback::Load => "load_error",
            Fallback::DiskWrite => "disk_write",
            Fallback::TpDigest => "tp_digest",
            Fallback::Delete => "delete_error",
        }
    }
    fn idx(self) -> usize {
        Fallback::ALL.iter().position(|&f| f == self).unwrap()
    }
}

/// Options of one record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub dir: Option<PathBuf>,
    pub cap_bytes: u64,
    pub read_only: bool,
}

struct Disk {
    dir: PathBuf,
    /// the file number of record chunk 0 (positions p live in file seq0 + p / CHUNK_IDS)
    seq0: u64,
    /// the newest chunk's file, open to append: (chunk number, file)
    file: Option<(u64, std::fs::File)>,
    /// held while this server writes the directory
    _lock: Option<std::fs::File>,
    /// persistence live (false: read only, locked, or a write failed)
    writes: bool,
}

/// The server's record. Owned by the scheduler thread (every TP rank holds its own).
pub struct CopyRecord {
    pub ids: Ids,
    pub rates: Rates,
    ident: Identity,
    cap_chunks: usize,
    read_only: bool,
    disk: Option<Disk>,
    /// ids this server added (overlaps included) / ids its requests read or wrote that were held already
    pub added: u64,
    pub held_dedup: u64,
    pub lookups: u64,
    pub proposals: u64,
    pub proposed: u64,
    pub accepted: u64,
    pub band_props: [u64; 3],
    pub band_proposed: [u64; 3],
    pub band_accepted: [u64; 3],
    pub priced_out: u64,
    pub probes: u64,
    pub quiet: u64,
    pub clears: u64,
    /// ids loaded from disk at boot
    pub loaded: u64,
}

/// Lines for the boot report, and the fallbacks (each counted once).
#[derive(Default, Debug)]
pub struct OpenNotes {
    pub lines: Vec<String>,
    pub fallbacks: Vec<Fallback>,
}

fn bucket_bits(cap_chunks: usize) -> u32 {
    let mut bits = 16u32;
    while bits < 28 && (1usize << bits) < cap_chunks * (CHUNK_IDS / 8) { bits += 1; }
    bits
}

fn chunk_path(dir: &Path, seq: u64) -> PathBuf {
    dir.join(format!("{seq:016x}.ids"))
}

fn open_0600(path: &Path, create_new: bool, append: bool) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut o = std::fs::OpenOptions::new();
    o.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    if create_new { o.write(true).create_new(true); } else if append { o.append(true); } else { o.read(true).write(true).create(true); }
    o.open(path)
}

impl CopyRecord {
    /// An in-memory record (no directory).
    pub fn memory(ident: Identity, cap_bytes: u64, read_only: bool) -> Self {
        let cap_chunks = ((cap_bytes / CHUNK_FILE_BYTES) as usize).max(2);
        CopyRecord {
            ids: Ids::new(bucket_bits(cap_chunks)), rates: Rates::default(), ident, cap_chunks, read_only, disk: None,
            added: 0, held_dedup: 0, lookups: 0, proposals: 0, proposed: 0, accepted: 0, band_props: [0; 3],
            band_proposed: [0; 3], band_accepted: [0; 3], priced_out: 0, probes: 0, quiet: 0, clears: 0, loaded: 0,
        }
    }

    /// Open the record: in memory, or from / to `o.dir`. Never fails: trouble degrades (memory only, or empty) with
    /// a line and a counted reason in the notes.
    pub fn open(ident: Identity, o: &Options) -> (Self, OpenNotes) {
        let mut r = CopyRecord::memory(ident, o.cap_bytes, o.read_only);
        let mut notes = OpenNotes::default();
        let Some(dir) = o.dir.clone() else { return (r, notes) };
        let note = |n: &mut OpenNotes, f: Fallback, line: String| { n.fallbacks.push(f); n.lines.push(line); };
        if let Err(e) = { use std::os::unix::fs::DirBuilderExt; std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir) } {
            note(&mut notes, Fallback::Dir, format!("cannot make the record directory {}: {e}; the record lives in memory for this run", dir.display()));
            return (r, notes);
        }
        // the identity: a record written under another tokenizer is never read and never written
        let idp = dir.join("identity");
        let want = ident.text();
        match std::fs::read_to_string(&idp) {
            Ok(got) if got == want => {}
            Ok(got) => {
                let other = got.lines().find_map(|l| l.strip_prefix("digest ")).map(|d| d.chars().take(16).collect::<String>())
                    .unwrap_or_else(|| "unknown".into());
                note(&mut notes, Fallback::Identity, format!("{} holds a record written under another tokenizer (identity {other}, this \
model's {}): it is NOT read and NOT written; this run starts with an EMPTY record in memory (name another --copy-span-record-dir to \
persist)", dir.display(), ident.short()));
                return (r, notes);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if !o.read_only {
                    let part = dir.join("identity.part");
                    let ok = open_0600(&part, false, false).and_then(|mut f| { f.set_len(0)?; f.write_all(want.as_bytes())?; f.sync_all() })
                        .and_then(|_| std::fs::rename(&part, &idp));
                    if let Err(e) = ok {
                        note(&mut notes, Fallback::Dir, format!("cannot write {}: {e}; the record lives in memory for this run", idp.display()));
                        return (r, notes);
                    }
                }
            }
            Err(e) => {
                note(&mut notes, Fallback::Dir, format!("cannot read {}: {e}; the record lives in memory for this run", idp.display()));
                return (r, notes);
            }
        }
        // the lock: a second server reads the record as it was and writes nothing
        let mut writes = !o.read_only;
        let mut lock = None;
        if writes {
            match open_0600(&dir.join("lock"), false, false) {
                Ok(f) => match f.try_lock() {
                    Ok(()) => lock = Some(f),
                    Err(_) => {
                        writes = false;
                        note(&mut notes, Fallback::Locked, format!("another server is writing {}: this one reads the record as it was \
and writes nothing to it", dir.display()));
                    }
                },
                Err(e) => {
                    writes = false;
                    note(&mut notes, Fallback::Locked, format!("cannot open {}/lock ({e}): the record is read and not written", dir.display()));
                }
            }
        }
        r.disk = Some(Disk { dir, seq0: 0, file: None, _lock: lock, writes });
        if let Err(e) = r.load() {
            note(&mut notes, Fallback::Load, format!("record load stopped early ({e}); what loaded before it is kept"));
        }
        r.loaded = r.ids.held();
        (r, notes)
    }

    fn header(&self) -> [u8; HEAD_BYTES] {
        let mut h = [0u8; HEAD_BYTES];
        h[..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&self.ident.vocab.to_le_bytes());
        h[16..32].copy_from_slice(&self.ident.digest[..16]);
        h
    }

    fn list_chunks(dir: &Path) -> std::io::Result<Vec<u64>> {
        let mut v = Vec::new();
        for e in std::fs::read_dir(dir)? {
            let name = e?.file_name();
            let name = name.to_string_lossy();
            if name.len() == 20 && name.ends_with(".ids") {
                if let Ok(seq) = u64::from_str_radix(&name[..16], 16) { v.push(seq); }
            }
        }
        v.sort_unstable();
        Ok(v)
    }

    /// Chunk files in order from the first: each full but the last; ids read up to the last mark; the files cut to
    /// what was read (when this server writes).
    fn load(&mut self) -> anyhow::Result<()> {
        let d = self.disk.as_ref().unwrap();
        let dir = d.dir.clone();
        let writes = d.writes;
        let seqs = Self::list_chunks(&dir)?;
        let Some(&seq0) = seqs.first() else { return Ok(()) };
        self.disk.as_mut().unwrap().seq0 = seq0;
        let head = self.header();
        let mut pending: Vec<u32> = Vec::new();
        let mut err = None;
        for (k, &seq) in seqs.iter().enumerate() {
            if seq != seq0 + k as u64 { break; } // a gap: the files past it go
            let mut f = match std::fs::File::open(chunk_path(&dir, seq)) { Ok(f) => f, Err(e) => { err = Some(e.into()); break; } };
            let mut buf = Vec::new();
            if let Err(e) = f.read_to_end(&mut buf) { err = Some(e.into()); break; }
            if buf.len() < HEAD_BYTES || buf[..HEAD_BYTES] != head {
                err = Some(anyhow::anyhow!("{} has a foreign header", chunk_path(&dir, seq).display()));
                break;
            }
            let count = (buf.len() - HEAD_BYTES) / 4;
            if count > CHUNK_IDS { err = Some(anyhow::anyhow!("{} is too long", chunk_path(&dir, seq).display())); break; }
            let mut bad = false;
            for i in 0..count {
                let o = HEAD_BYTES + 4 * i;
                let t = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
                if t == MARK {
                    self.ids.add(&pending);
                    pending.clear();
                } else if t < self.ident.vocab {
                    pending.push(t);
                } else {
                    bad = true;
                    break;
                }
            }
            if bad { err = Some(anyhow::anyhow!("{} holds an id outside the vocabulary", chunk_path(&dir, seq).display())); break; }
            if count < CHUNK_IDS { break; } // a short file is the last
        }
        if writes { self.cut(&seqs); }
        while self.ids.chunks() > self.cap_chunks { self.drop_chunk(); }
        match err { Some(e) => Err(e), None => Ok(()) }
    }

    /// The files cut to what the record read: the last one's torn tail truncated, the files after it deleted.
    fn cut(&mut self, seqs: &[u64]) {
        let d = self.disk.as_ref().unwrap();
        let chunks = (self.ids.end as usize).div_ceil(CHUNK_IDS);
        for (k, &seq) in seqs.iter().enumerate() {
            let p = chunk_path(&d.dir, seq);
            if k >= chunks || seq != d.seq0 + k as u64 {
                let _ = std::fs::remove_file(&p);
            } else if k + 1 == chunks {
                let keep = HEAD_BYTES as u64 + 4 * (self.ids.end - (k * CHUNK_IDS) as u64);
                if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&p) { let _ = f.set_len(keep); }
            }
        }
    }

    /// The oldest chunk freed and its file deleted.
    fn drop_chunk(&mut self) {
        if !self.ids.drop_oldest() { return; }
        if let Some(d) = self.disk.as_mut() {
            let seq = d.seq0 + self.ids.first - 1;
            if d.file.as_ref().is_some_and(|(k, _)| *k == self.ids.first - 1) { d.file = None; }
            if d.writes { let _ = std::fs::remove_file(chunk_path(&d.dir, seq)); }
        }
    }

    pub fn read_only(&self) -> bool {
        self.read_only
    }
    pub fn persisting(&self) -> bool {
        self.disk.as_ref().is_some_and(|d| d.writes)
    }
    pub fn dir(&self) -> Option<&Path> {
        self.disk.as_ref().map(|d| d.dir.as_path())
    }
    pub fn cap_chunks(&self) -> usize {
        self.cap_chunks
    }
    /// Bytes the held ids take on disk (headers included).
    pub fn disk_bytes(&self) -> u64 {
        if self.disk.is_none() { return 0; }
        self.ids.chunks() as u64 * HEAD_BYTES as u64 + 4 * self.ids.held()
    }

    /// TP: stop persisting for the boot and drop everything held (the ranks loaded different records). The files
    /// stay as they are.
    pub fn forget_for_boot(&mut self) {
        self.ids.clear();
        self.loaded = 0;
        self.rates = Rates::default();
        if let Some(d) = self.disk.as_mut() { d.writes = false; d.file = None; }
    }

    /// A request's prompt (the ids not already held). Nothing in read-only mode.
    pub fn add_prompt(&mut self, prompt: &[u32]) -> Option<Fallback> {
        if self.read_only { return None; }
        self.add_dedup(prompt)
    }

    /// A finished reply after the last OVERLAP ids of its prompt (`tail`).
    pub fn add_reply(&mut self, tail: &[u32], reply: &[u32]) -> Option<Fallback> {
        if self.read_only || reply.is_empty() { return None; }
        let mut seg = Vec::with_capacity(tail.len() + reply.len());
        seg.extend_from_slice(&tail[tail.len().saturating_sub(OVERLAP)..]);
        seg.extend_from_slice(reply);
        self.add_dedup(&seg)
    }

    /// The runs of `seg` the record lacks, each with its overlaps. Returns a fallback when persistence just failed.
    fn add_dedup(&mut self, seg: &[u32]) -> Option<Fallback> {
        let mut fresh = 0usize;
        let mut i = 0usize;
        let mut fb = None;
        while i + KEY_LEN <= seg.len() {
            let (len, at) = self.ids.held_run(&seg[i..]);
            if len >= DEDUP_STRIDE {
                let (mut j, mut q) = (i, at);
                while j > fresh && q > self.ids.base() && self.ids.id(q - 1) == seg[j - 1] {
                    j -= 1;
                    q -= 1;
                }
                if i + len - j >= DEDUP_RUN {
                    fb = fb.or(self.put(seg, fresh, j));
                    self.held_dedup += (i + len - j) as u64;
                    fresh = i + len;
                    i = fresh;
                    continue;
                }
            }
            i += DEDUP_STRIDE;
        }
        fb.or(self.put(seg, fresh, seg.len()))
    }

    /// `seg[from..to]` with its overlaps, added (and written when persisting).
    fn put(&mut self, seg: &[u32], from: usize, to: usize) -> Option<Fallback> {
        if to <= from { return None; }
        let piece = &seg[from.saturating_sub(OVERLAP)..(to + OVERLAP).min(seg.len())];
        if piece.len() < KEY_LEN || piece.iter().any(|&t| t >= self.ident.vocab) { return None; }
        let start = self.ids.end;
        self.ids.add(piece);
        self.added += piece.len() as u64;
        let mut fb = None;
        if self.persisting() {
            if let Err(e) = self.write(start, self.ids.end) {
                let d = self.disk.as_mut().unwrap();
                d.writes = false;
                d.file = None;
                crate::reprintln!("[exl3-serve] copy-span-record: writing {} failed ({e}); the record stops persisting for this run \
(what it holds in memory is unchanged)", d.dir.display());
                fb = Some(Fallback::DiskWrite);
            }
        }
        while self.ids.chunks() > self.cap_chunks { self.drop_chunk(); }
        fb
    }

    /// Positions [start, end) appended to their chunk files.
    fn write(&mut self, start: u64, end: u64) -> std::io::Result<()> {
        let head = self.header();
        let mut p = start;
        let mut bytes = Vec::with_capacity(((end - start) as usize).min(1 << 16) * 4);
        while p < end {
            let k = p / CHUNK_IDS as u64;
            let stop = end.min((k + 1) * CHUNK_IDS as u64);
            let d = self.disk.as_mut().unwrap();
            if !d.file.as_ref().is_some_and(|(c, _)| *c == k) {
                d.file = None;
                let path = chunk_path(&d.dir, d.seq0 + k);
                let f = match open_0600(&path, false, true) {
                    Ok(mut f) => { f.seek(SeekFrom::End(0))?; f }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        let mut f = open_0600(&path, true, false)?;
                        f.write_all(&head)?;
                        f
                    }
                    Err(e) => return Err(e),
                };
                d.file = Some((k, f));
            }
            bytes.clear();
            for q in p..stop { bytes.extend_from_slice(&self.ids.id(q).to_le_bytes()); }
            self.disk.as_mut().unwrap().file.as_mut().unwrap().1.write_all(&bytes)?;
            p = stop;
        }
        Ok(())
    }

    /// Empty it: every chunk file deleted, ids, rates and counters reset (identity and lock stay). Err names a file
    /// that could not be removed (persistence then stops for the boot).
    pub fn empty(&mut self) -> Result<(), String> {
        let next = self.disk.as_ref().map_or(0, |d| d.seq0 + self.ids.end.div_ceil(CHUNK_IDS as u64) + 1);
        let mut problem = None;
        if let Some(d) = self.disk.as_mut() {
            d.file = None;
            if d.writes {
                let seqs = Self::list_chunks(&d.dir).unwrap_or_default();
                for &seq in &seqs {
                    let p = chunk_path(&d.dir, seq);
                    if let Err(e) = std::fs::remove_file(&p) {
                        if e.kind() != std::io::ErrorKind::NotFound && problem.is_none() {
                            problem = Some(format!("{} could not be removed ({e})", p.display()));
                        }
                    }
                }
                d.seq0 = next.max(seqs.last().map_or(0, |s| s + 1));
                if problem.is_some() { d.writes = false; }
            }
        }
        self.ids.clear();
        self.rates = Rates::default();
        self.added = 0;
        self.held_dedup = 0;
        self.lookups = 0;
        self.proposals = 0;
        self.proposed = 0;
        self.accepted = 0;
        self.band_props = [0; 3];
        self.band_proposed = [0; 3];
        self.band_accepted = [0; 3];
        self.priced_out = 0;
        self.probes = 0;
        self.quiet = 0;
        self.loaded = 0;
        self.clears += 1;
        match problem { Some(p) => Err(p), None => Ok(()) }
    }

    /// Ask the record for a copy: the tail of `ctx` must match at least `min_n` ids; `width` = the ids the copy's
    /// first round would offer; `head_q16` = the tokens a round of the request's own drafts emits lately (x65536).
    pub fn offer(&mut self, ctx: &[u32], width: usize, head_q16: i64, min_n: usize, rd: &mut Reader) -> Option<Offer> {
        if width == 0 || ctx.len() < min_n.max(KEY_LEN) || self.ids.held() == 0 { return None; }
        if rd.rounds < rd.quiet_until {
            self.quiet += 1;
            return None;
        }
        // the price first: the record is walked only when a copy can pay, or for a probe
        let mut probing = false;
        if !self.rates.pays(width, head_q16) {
            probing = self.rates.probe();
            if !probing {
                self.priced_out += 1;
                return None;
            }
        }
        self.lookups += 1;
        let m = self.ids.longest(ctx);
        if m.len < min_n.max(KEY_LEN) { return None; }
        let snap = self.ids.copy(m.at, SNAP_MAX);
        if snap.is_empty() { return None; }
        let b = band(m.len);
        if !probing && expected_q16(self.rates.rate_q16(b), width.min(snap.len())) <= head_q16 {
            probing = self.rates.probe();
            if !probing {
                self.priced_out += 1;
                return None;
            }
        }
        self.probes += probing as u64;
        Some(Offer { snap, match_len: m.len, band: b, probing })
    }

    /// Count an offer the proposer turned into a proposal of `n` ids (band `b`).
    pub fn count_proposal(&mut self, b: usize, n: usize) {
        self.proposals += 1;
        self.band_props[b] += 1;
        self.proposed += n as u64;
        self.band_proposed[b] += n as u64;
    }

    /// A request's record-copy rounds since its last ask (in round order).
    pub fn observe(&mut self, obs: &[Obs]) {
        for o in obs {
            self.rates.observe(o.band, o.proposed, o.accepted, o.probing);
            self.accepted += o.accepted as u64;
            self.band_accepted[o.band] += o.accepted as u64;
        }
    }

    /// The `--copy-span-record` boot line's state and size part.
    pub fn describe(&self) -> String {
        let where_ = match &self.disk {
            None => "in memory only (no --copy-span-record-dir: it lives as long as the server)".to_string(),
            Some(d) if d.writes => format!("persisted in {}", d.dir.display()),
            Some(d) => format!("read from {} and NOT written there", d.dir.display()),
        };
        let held = self.ids.held();
        format!("{}{} ids held ({:.1} MiB in memory{}); cap {} MiB ({} chunks of {} ids, oldest dropped first); {}",
                if held == 0 { "EMPTY: " } else { "POPULATED: " }, held, self.ids.ram_bytes() as f64 / 1048576.0,
                if self.disk.is_some() { format!(", {:.1} MiB on disk", self.disk_bytes() as f64 / 1048576.0) } else { String::new() },
                self.cap_chunks as u64 * CHUNK_FILE_BYTES / 1048576, self.cap_chunks, CHUNK_IDS, where_)
    }
}

// ---------------------------------------------------------------------------------------------
// Boot resolution, counters (`velogb10_copy_span_record_*`), the HTTP view
// ---------------------------------------------------------------------------------------------

/// How `--copy-span-record` resolved on this boot (the `velogb10_copy_span_record_state` gauge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    NotAsked = 0,
    On = 1,
    /// asked, but the copy arm is not armed on this boot (`--copy-span off`, or it runs off): the record is off
    OffNoCopySpan = 2,
}

/// Pure boot resolution. Never a refusal.
pub fn resolve(asked: bool, copy_armed: bool) -> State {
    if !asked { State::NotAsked } else if !copy_armed { State::OffNoCopySpan } else { State::On }
}

static ST: AtomicU64 = AtomicU64::new(0);
static RO: AtomicU64 = AtomicU64::new(0);
static PERSIST: AtomicU64 = AtomicU64::new(0);
static HELD: AtomicU64 = AtomicU64::new(0);
static RAM: AtomicU64 = AtomicU64::new(0);
static DISK: AtomicU64 = AtomicU64::new(0);
static CAP: AtomicU64 = AtomicU64::new(0);
static LOADED: AtomicU64 = AtomicU64::new(0);
static ADDED: AtomicU64 = AtomicU64::new(0);
static DEDUP: AtomicU64 = AtomicU64::new(0);
static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static PROPS: AtomicU64 = AtomicU64::new(0);
static PROPOSED: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);
static PRICED_OUT: AtomicU64 = AtomicU64::new(0);
static PROBES: AtomicU64 = AtomicU64::new(0);
static QUIET: AtomicU64 = AtomicU64::new(0);
static CLEARS: AtomicU64 = AtomicU64::new(0);
static PROBE_EVERY: AtomicU64 = AtomicU64::new(PROBE_FIRST as u64);
static HOST_US: AtomicU64 = AtomicU64::new(0);
static BAND_PROPS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static BAND_PROPOSED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static BAND_ACCEPTED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static BAND_RATE: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static FALLBACKS: [AtomicU64; 8] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                                    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static CLEAR_REQ: AtomicBool = AtomicBool::new(false);

pub fn set_state(st: State, read_only: bool) {
    ST.store(st as u64, Relaxed);
    RO.store(read_only as u64, Relaxed);
    if st == State::OffNoCopySpan { count_fallback(Fallback::NoCopySpan); }
}
pub fn state() -> u64 {
    ST.load(Relaxed)
}
pub fn count_fallback(f: Fallback) {
    FALLBACKS[f.idx()].fetch_add(1, Relaxed);
}
pub fn add_host_us(us: u64) {
    HOST_US.fetch_add(us, Relaxed);
}

/// Mirror the record's state into the counters (the scheduler thread calls it after every change).
pub fn publish(r: &CopyRecord) {
    PERSIST.store(r.persisting() as u64, Relaxed);
    HELD.store(r.ids.held(), Relaxed);
    RAM.store(r.ids.ram_bytes(), Relaxed);
    DISK.store(r.disk_bytes(), Relaxed);
    CAP.store(r.cap_chunks as u64 * CHUNK_FILE_BYTES, Relaxed);
    LOADED.store(r.loaded, Relaxed);
    ADDED.store(r.added, Relaxed);
    DEDUP.store(r.held_dedup, Relaxed);
    LOOKUPS.store(r.lookups, Relaxed);
    PROPS.store(r.proposals, Relaxed);
    PROPOSED.store(r.proposed, Relaxed);
    ACCEPTED.store(r.accepted, Relaxed);
    PRICED_OUT.store(r.priced_out, Relaxed);
    PROBES.store(r.probes, Relaxed);
    QUIET.store(r.quiet, Relaxed);
    CLEARS.store(r.clears, Relaxed);
    PROBE_EVERY.store(r.rates.probe_every as u64, Relaxed);
    for b in 0..3 {
        BAND_PROPS[b].store(r.band_props[b], Relaxed);
        BAND_PROPOSED[b].store(r.band_proposed[b], Relaxed);
        BAND_ACCEPTED[b].store(r.band_accepted[b], Relaxed);
        BAND_RATE[b].store(r.rates.rate_q16(b) as u64, Relaxed);
    }
}

/// `DELETE /v1/copy-record`: Ok = the scheduler empties the record before it admits the next request (and on every
/// TP rank, shipped with the next step); Err(409 text) = nothing to empty or not allowed.
pub fn request_clear() -> Result<(), &'static str> {
    if ST.load(Relaxed) != State::On as u64 { return Err("the copy record is not on on this server (--copy-span-record on)"); }
    if RO.load(Relaxed) != 0 { return Err("the copy record is read-only on this server (--copy-span-record-read-only on)"); }
    CLEAR_REQ.store(true, Relaxed);
    Ok(())
}

/// The scheduler: was an empty requested since the last call?
pub fn take_clear_request() -> bool {
    CLEAR_REQ.swap(false, Relaxed)
}

fn band_name(b: usize) -> &'static str {
    ["16_31", "32_63", "64_up"][b]
}

/// `GET /v1/copy-record`.
pub fn state_json() -> serde_json::Value {
    let st = match ST.load(Relaxed) { 1 => "on", 2 => "off_no_copy_span", _ => "not_asked" };
    const NAMES: [&str; 3] = ["16-31", "32-63", "64+"];
    let bands: Vec<serde_json::Value> = (0..3).map(|b| serde_json::json!({
        "match_len": NAMES[b],
        "landing_rate": BAND_RATE[b].load(Relaxed) as f64 / Q16 as f64,
        "proposals": BAND_PROPS[b].load(Relaxed),
        "proposed_tokens": BAND_PROPOSED[b].load(Relaxed),
        "accepted_tokens": BAND_ACCEPTED[b].load(Relaxed),
    })).collect();
    serde_json::json!({
        "object": "copy_record",
        "state": st,
        "read_only": RO.load(Relaxed) != 0,
        "persisting": PERSIST.load(Relaxed) != 0,
        "ids_held": HELD.load(Relaxed),
        "memory_bytes": RAM.load(Relaxed),
        "disk_bytes": DISK.load(Relaxed),
        "cap_bytes": CAP.load(Relaxed),
        "ids_loaded_at_boot": LOADED.load(Relaxed),
        "ids_added": ADDED.load(Relaxed),
        "ids_already_held": DEDUP.load(Relaxed),
        "lookups": LOOKUPS.load(Relaxed),
        "proposals": PROPS.load(Relaxed),
        "proposed_tokens": PROPOSED.load(Relaxed),
        "accepted_tokens": ACCEPTED.load(Relaxed),
        "bands": bands,
        "empty_pending": CLEAR_REQ.load(Relaxed),
        "emptied": CLEARS.load(Relaxed),
    })
}

pub fn metrics_append(o: &mut String) {
    let g = |o: &mut String, name: &str, help: &str, kind: &str, v: String| {
        o.push_str(&format!("# HELP velogb10_copy_span_record_{name} {help}\n# TYPE velogb10_copy_span_record_{name} {kind}\nvelogb10_copy_span_record_{name} {v}\n"));
    };
    g(o, "state", "--copy-span-record on this boot: 0 = not asked, 1 = on, 2 = asked but off (--copy-span not armed).", "gauge", ST.load(Relaxed).to_string());
    g(o, "read_only", "--copy-span-record-read-only (1 = the record is read, never added to).", "gauge", RO.load(Relaxed).to_string());
    g(o, "persisting", "1 = the record is written to --copy-span-record-dir as it grows.", "gauge", PERSIST.load(Relaxed).to_string());
    g(o, "ids_held", "Token ids the record holds (segment marks included). A populated record makes repeated prompts faster than on a fresh server.", "gauge", HELD.load(Relaxed).to_string());
    g(o, "memory_bytes", "Memory the record takes (ids, links, key table).", "gauge", RAM.load(Relaxed).to_string());
    g(o, "disk_bytes", "Bytes the held ids take in --copy-span-record-dir (0 without a directory).", "gauge", DISK.load(Relaxed).to_string());
    g(o, "cap_bytes", "--copy-span-record-mib as chunk files (the oldest chunk is dropped past it).", "gauge", CAP.load(Relaxed).to_string());
    g(o, "ids_loaded", "Ids loaded from --copy-span-record-dir at boot (0 after an empty, or when the TP ranks disagreed).", "gauge", LOADED.load(Relaxed).to_string());
    g(o, "ids_added_total", "Ids this server added to the record (prompts and replies, with their overlaps).", "counter", ADDED.load(Relaxed).to_string());
    g(o, "ids_already_held_total", "Ids of prompts and replies not added because the record held them already (runs of 128 or more).", "counter", DEDUP.load(Relaxed).to_string());
    g(o, "lookups_total", "Record lookups (a request's own context had nothing to copy and a copy could pay, or a probe).", "counter", LOOKUPS.load(Relaxed).to_string());
    g(o, "proposals_total", "Copies started from the record (a copy's first round; its later rounds are cursor rounds).", "counter", PROPS.load(Relaxed).to_string());
    g(o, "proposed_tokens_total", "Record-copied tokens offered to the verify, by copies' first rounds only (later rounds: velogb10_copy_span_offered_tokens_total).", "counter", PROPOSED.load(Relaxed).to_string());
    g(o, "accepted_tokens_total", "Record-copied tokens the verify accepted (every round of a record copy).", "counter", ACCEPTED.load(Relaxed).to_string());
    g(o, "priced_out_total", "Asks the landing rates priced out (a record copy was not expected to beat the request's own drafts).", "counter", PRICED_OUT.load(Relaxed).to_string());
    g(o, "probes_total", "Record copies offered as probes although priced out.", "counter", PROBES.load(Relaxed).to_string());
    g(o, "quiet_total", "Asks skipped because the request's last record copy landed nothing (it leaves the record alone for 4 rounds).", "counter", QUIET.load(Relaxed).to_string());
    g(o, "probe_every", "Priced-out asks between probes (32, doubling to 1024 while probes miss).", "gauge", PROBE_EVERY.load(Relaxed).to_string());
    g(o, "emptied_total", "Times the record was emptied (DELETE /v1/copy-record).", "counter", CLEARS.load(Relaxed).to_string());
    g(o, "host_seconds_total", "Host time of the record (adds, lookups) on the scheduler thread.", "counter", format!("{:.6}", HOST_US.load(Relaxed) as f64 / 1e6));
    o.push_str("# HELP velogb10_copy_span_record_landing_rate Chance that a record-copied id lands when the ids before it did, by match-length band (moving, starts at the priors 0.6 / 0.8 / 0.9).\n# TYPE velogb10_copy_span_record_landing_rate gauge\n");
    for b in 0..3 {
        o.push_str(&format!("velogb10_copy_span_record_landing_rate{{band=\"{}\"}} {:.4}\n", band_name(b), BAND_RATE[b].load(Relaxed) as f64 / Q16 as f64));
    }
    o.push_str("# HELP velogb10_copy_span_record_band_proposals_total Record copies started, by match-length band.\n# TYPE velogb10_copy_span_record_band_proposals_total counter\n");
    for b in 0..3 {
        o.push_str(&format!("velogb10_copy_span_record_band_proposals_total{{band=\"{}\"}} {}\n", band_name(b), BAND_PROPS[b].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_copy_span_record_band_accepted_tokens_total Record-copied tokens accepted, by match-length band.\n# TYPE velogb10_copy_span_record_band_accepted_tokens_total counter\n");
    for b in 0..3 {
        o.push_str(&format!("velogb10_copy_span_record_band_accepted_tokens_total{{band=\"{}\"}} {}\n", band_name(b), BAND_ACCEPTED[b].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_copy_span_record_fallback_total Asked for but served with less, by reason (copy_span_off: the record is off; tokenizer_mismatch: the directory belongs to another tokenizer, run empty and unwritten; dir_error / locked / load_error / disk_write / delete_error: persistence reduced, memory unchanged; tp_digest: the TP ranks loaded different records, all started empty).\n# TYPE velogb10_copy_span_record_fallback_total counter\n");
    for f in Fallback::ALL {
        o.push_str(&format!("velogb10_copy_span_record_fallback_total{{reason=\"{}\"}} {}\n", f.reason(), FALLBACKS[f.idx()].load(Relaxed)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(v: u32) -> Identity {
        Identity::from_tokenizer_bytes(b"{\"model\":\"test\"}", v)
    }
    fn mem() -> CopyRecord {
        CopyRecord::memory(ident(1_000_000), 8 * CHUNK_FILE_BYTES, false)
    }
    fn uniq(base: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| base + i).collect()
    }
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }
    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("copyrec_test_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn record_longest_match_and_copy() {
        let mut r = mem();
        let a = uniq(100, 200);
        r.add_prompt(&a);
        // a tail of 40 ids of `a` matches 40 back (bounded by the context), continuation follows it
        let ctx: Vec<u32> = [uniq(5000, 10), a[60..100].to_vec()].concat();
        let m = r.ids.longest(&ctx);
        assert_eq!(m.len, 40);
        assert_eq!(r.ids.copy(m.at, 5), a[100..105].to_vec());
        // a longer context extends to MAX_MATCH
        let m = r.ids.longest(&a[..150]);
        assert_eq!(m.len, MAX_MATCH);
        assert_eq!(r.ids.copy(m.at, 1000), a[150..].to_vec(), "a copy stops at the segment's mark");
        // the newest of two equal matches wins; a longer older one beats a shorter newer one
        let b: Vec<u32> = [uniq(9000, 30), a[140..150].to_vec(), vec![7]].concat();
        r.add_prompt(&b);
        let m = r.ids.longest(&a[140..150]);
        assert_eq!((m.len, r.ids.id(m.at)), (10, 7), "newest of equal length");
        let m = r.ids.longest(&a[100..150]);
        assert_eq!((m.len, r.ids.id(m.at)), (50, a[150]), "the longer match wins");
        // a tail shorter than a key, or unknown, finds nothing; a match that ends at a mark is skipped
        assert_eq!(r.ids.longest(&a[..5]).len, 0);
        assert_eq!(r.ids.longest(&uniq(70_000, 30)).len, 0);
        assert_eq!(r.ids.longest(&a[150..200]).len, 0, "the end of a segment has no continuation");
    }

    #[test]
    fn record_dedup_skips_long_held_runs_and_keeps_short_ones() {
        let mut r = mem();
        let a = uniq(100, 1000);
        r.add_prompt(&a);
        let after_first = r.ids.held();
        assert_eq!(after_first, 1001);
        // the same prompt again: nothing new
        r.add_prompt(&a);
        assert_eq!(r.ids.held(), after_first, "a request sent twice adds nothing the second time");
        assert_eq!(r.held_dedup, 1000);
        // a prompt that is `a` + 300 new ids: only the new ids (with one overlap before them) are added
        let b: Vec<u32> = [a.clone(), uniq(50_000, 300)].concat();
        r.add_prompt(&b);
        let added = r.ids.held() - after_first;
        assert!(added >= 300 && added <= 300 + OVERLAP as u64 + 1 + DEDUP_STRIDE as u64, "added {added}");
        // a held run shorter than DEDUP_RUN is written again
        let short: Vec<u32> = [uniq(80_000, 50), a[0..100].to_vec(), uniq(90_000, 50)].concat();
        let before = r.ids.held();
        r.add_prompt(&short);
        assert_eq!(r.ids.held() - before, short.len() as u64 + 1);
        // the reply after the prompt's tail: found across the join
        let reply = uniq(200_000, 64);
        r.add_reply(&a, &reply);
        let m = r.ids.longest(&a[900..1000]);
        assert_eq!(r.ids.copy(m.at, 4), reply[..4].to_vec(), "the prompt tail continues into the reply");
        // read-only adds nothing
        let mut ro = CopyRecord::memory(ident(1_000_000), 8 * CHUNK_FILE_BYTES, true);
        ro.add_prompt(&a);
        ro.add_reply(&a, &reply);
        assert_eq!(ro.ids.held(), 0);
    }

    #[test]
    fn record_drops_the_oldest_chunk_past_the_cap() {
        let mut r = CopyRecord::memory(ident(u32::MAX - 1), 2 * CHUNK_FILE_BYTES, false);
        assert_eq!(r.cap_chunks(), 2);
        let first = uniq(1, 1000);
        r.add_prompt(&first);
        let mut rng = Lcg(5);
        // fill past two chunks with distinct segments
        let mut n = 0;
        while r.ids.end() < 3 * CHUNK_IDS as u64 {
            let seg: Vec<u32> = (0..4000).map(|_| 2_000_000 + rng.next() % 1_000_000_000).collect();
            r.add_prompt(&seg);
            n += 1;
        }
        assert!(n > 500);
        assert_eq!(r.ids.chunks(), 2, "never more chunks than the cap");
        assert!(r.ids.base() >= CHUNK_IDS as u64);
        assert_eq!(r.ids.longest(&first[..100]).len, 0, "the oldest ids are gone");
        // lookups that reach links into the dropped chunk end there (no panic)
        for _ in 0..200 {
            let ctx: Vec<u32> = (0..20).map(|_| rng.next() % 50).collect();
            let _ = r.ids.longest(&ctx);
            let _ = r.ids.held_run(&ctx);
        }
    }

    #[test]
    fn record_persists_and_round_trips() {
        let dir = tmpdir("rt");
        let o = Options { dir: Some(dir.clone()), cap_bytes: 8 * CHUNK_FILE_BYTES, read_only: false };
        let a = uniq(100, 3000);
        let reply = uniq(10_000, 500);
        let digest = {
            let (mut r, notes) = CopyRecord::open(ident(1_000_000), &o);
            assert!(notes.fallbacks.is_empty(), "{:?}", notes.lines);
            assert!(r.persisting());
            r.add_prompt(&a);
            r.add_reply(&a, &reply);
            r.ids.digest()
        };
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
            assert_eq!(std::fs::metadata(dir.join("identity")).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(std::fs::metadata(chunk_path(&dir, 0)).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // a restart reads the same record: same digest, the reply is found
        let (mut r2, notes) = CopyRecord::open(ident(1_000_000), &o);
        assert!(notes.fallbacks.is_empty(), "{:?}", notes.lines);
        assert_eq!(r2.ids.digest(), digest);
        assert_eq!(r2.loaded, r2.ids.held());
        let m = r2.ids.longest(&a[2900..]);
        assert_eq!(r2.ids.copy(m.at, 3), reply[..3].to_vec());
        // a torn tail (a segment without its mark) is cut at load
        r2.add_prompt(&uniq(400_000, 100));
        let full = r2.ids.digest();
        drop(r2);
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(chunk_path(&dir, 0)).unwrap();
            f.write_all(&[1u8, 0, 0, 0, 2, 0, 0, 0]).unwrap();
        }
        let (r3, _) = CopyRecord::open(ident(1_000_000), &o);
        assert_eq!(r3.ids.digest(), full, "the unmarked tail is not read");
        drop(r3);
        let len = std::fs::metadata(chunk_path(&dir, 0)).unwrap().len();
        assert_eq!(len, HEAD_BYTES as u64 + 4 * full_end(&dir, &o), "and the file is cut to what was read");
        // empty: files gone, record empty, and the next adds start a new file
        let (mut r4, _) = CopyRecord::open(ident(1_000_000), &o);
        assert!(r4.empty().is_ok());
        assert_eq!(r4.ids.held(), 0);
        assert!(CopyRecord::list_chunks(&dir).unwrap().is_empty());
        r4.add_prompt(&uniq(7, 50));
        drop(r4);
        let (mut r5, _) = CopyRecord::open(ident(1_000_000), &o);
        assert_eq!(r5.ids.held(), 51);
        // TP digest mismatch: empty, not persisting, nothing reported as loaded
        r5.forget_for_boot();
        assert_eq!((r5.ids.held(), r5.loaded, r5.persisting()), (0, 0, false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn full_end(dir: &Path, o: &Options) -> u64 {
        let (r, _) = CopyRecord::open(ident(1_000_000), &Options { dir: Some(dir.to_path_buf()), read_only: true, ..o.clone() });
        r.ids.end()
    }

    #[test]
    fn record_other_tokenizer_is_never_read_or_written() {
        let dir = tmpdir("ident");
        let o = Options { dir: Some(dir.clone()), cap_bytes: 8 * CHUNK_FILE_BYTES, read_only: false };
        {
            let (mut r, _) = CopyRecord::open(ident(1_000_000), &o);
            r.add_prompt(&uniq(100, 500));
        }
        let before = std::fs::read(chunk_path(&dir, 0)).unwrap();
        let other = Identity::from_tokenizer_bytes(b"{\"model\":\"another\"}", 1_000_000);
        let (mut r, notes) = CopyRecord::open(other, &o);
        assert_eq!(notes.fallbacks, vec![Fallback::Identity]);
        assert!(notes.lines[0].contains("another tokenizer") && notes.lines[0].contains("EMPTY"));
        assert_eq!(r.ids.held(), 0);
        assert!(!r.persisting());
        r.add_prompt(&uniq(900, 500));
        assert_eq!(std::fs::read(chunk_path(&dir, 0)).unwrap(), before, "nothing written to the other tokenizer's record");
        // a different vocabulary size is another identity too
        let (_, notes) = CopyRecord::open(ident(999), &o);
        assert_eq!(notes.fallbacks, vec![Fallback::Identity]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_second_server_reads_and_does_not_write() {
        let dir = tmpdir("lock");
        let o = Options { dir: Some(dir.clone()), cap_bytes: 8 * CHUNK_FILE_BYTES, read_only: false };
        let (mut a, _) = CopyRecord::open(ident(1_000_000), &o);
        a.add_prompt(&uniq(100, 500));
        let (b, notes) = CopyRecord::open(ident(1_000_000), &o);
        assert_eq!(notes.fallbacks, vec![Fallback::Locked]);
        assert!(!b.persisting());
        assert_eq!(b.ids.digest(), a.ids.digest());
        drop(a);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rates_band_gating_and_back_off() {
        assert_eq!((band(16), band(31), band(32), band(63), band(64), band(8)), (0, 0, 1, 1, 2, 0));
        let r = Rates::default();
        // priors: 0.6 / 0.8 / 0.9
        assert_eq!(r.rate_q16(0), 192 * Q16 / 320);
        assert!((r.rate_q16(2) as f64 / Q16 as f64 - 0.9).abs() < 1e-3);
        // a 4-id copy at 0.9 emits ~4.1 tokens; at 0.6 ~2.3
        let e2 = expected_q16(r.rate_q16(2), 4) as f64 / Q16 as f64;
        let e0 = expected_q16(r.rate_q16(0), 4) as f64 / Q16 as f64;
        assert!((e2 - 4.0951).abs() < 0.01 && (e0 - 2.3056).abs() < 0.01, "{e2} {e0}");
        assert!(r.pays(4, 3 * Q16) && !r.pays(4, 5 * Q16));
        // misses push a band's rate down; landings push it up
        let mut m = Rates::default();
        for _ in 0..40 { m.observe(2, 4, 0, false); }
        assert!(m.rate_q16(2) < Q16 / 10, "a band whose copies miss stops paying: {}", m.rate_q16(2));
        assert!(!m.pays(4, 2 * Q16) || m.rate_q16(1) > Q16 / 2);
        let mut l = Rates::default();
        for _ in 0..40 { l.observe(0, 4, 4, false); }
        assert!(l.rate_q16(0) > Q16 * 95 / 100);
        // probes: every 32nd priced-out ask; a missing probe doubles the interval (to 1024), a landing one resets it
        let mut p = Rates::default();
        let mut asks = 0;
        while !p.probe() { asks += 1; }
        assert_eq!(asks, 31);
        p.observe(0, 4, 1, true);
        assert_eq!(p.probe_every, 64);
        for _ in 0..10 { p.observe(0, 4, 0, true); }
        assert_eq!(p.probe_every, PROBE_MOST);
        p.observe(0, 4, 2, true);
        assert_eq!(p.probe_every, PROBE_FIRST);
    }

    #[test]
    fn offer_respects_min_n_price_and_quiet() {
        let mut r = mem();
        let a = uniq(100, 400);
        r.add_reply(&[], &a);
        let mut rd = Reader::default();
        // a 64-id match at the 0.9 prior pays against a head of 2 tokens a round
        let o = r.offer(&a[..100], 4, 2 * Q16, 16, &mut rd).expect("pays");
        assert_eq!((o.match_len, o.band, o.probing), (64, 2, false));
        assert_eq!(&o.snap[..3], &a[100..103]);
        // below min_n: nothing
        let ctx: Vec<u32> = [uniq(9000, 20), a[0..12].to_vec()].concat();
        assert!(r.offer(&ctx, 4, 2 * Q16, 16, &mut rd).is_none());
        // a head the record cannot beat: priced out (no lookup), except a probe every 32nd ask
        let looks = r.lookups;
        let got: Vec<bool> = (0..64).map(|_| r.offer(&a[..100], 4, 6 * Q16, 16, &mut rd).is_some()).collect();
        assert_eq!(got.iter().filter(|&&x| x).count(), 2, "two probes in 64 priced-out asks");
        assert_eq!(r.lookups - looks, 2);
        // quiet rounds: the reader leaves the record alone
        rd.quiet_until = rd.rounds + 3;
        assert!(r.offer(&a[..100], 4, 2 * Q16, 16, &mut rd).is_none());
        assert!(r.quiet > 0);
        // an empty record: nothing, nothing counted
        let mut e = mem();
        assert!(e.offer(&a[..100], 4, 2 * Q16, 16, &mut Reader::default()).is_none());
        assert_eq!(e.lookups + e.priced_out, 0);
    }

    /// Two instances fed the same stream of adds, offers and observations stay identical (the TP requirement).
    #[test]
    fn record_deterministic_two_instances() {
        let mut a = mem();
        let mut b = mem();
        let mut rng = Lcg(42);
        let mut rda = Reader::default();
        let mut rdb = Reader::default();
        let pool: Vec<Vec<u32>> = (0..8).map(|i| (0..300).map(|_| 1000 * i + rng.next() % 900).collect()).collect();
        for step in 0..600 {
            let k = (rng.next() % 8) as usize;
            match rng.next() % 4 {
                0 => { a.add_prompt(&pool[k]); b.add_prompt(&pool[k]); }
                1 => {
                    let cut = 20 + (rng.next() % 200) as usize;
                    a.add_reply(&pool[k][..cut], &pool[(k + 1) % 8]);
                    b.add_reply(&pool[k][..cut], &pool[(k + 1) % 8]);
                }
                _ => {
                    let end = 20 + (rng.next() % 250) as usize;
                    let head = (2 + rng.next() % 4) as i64 * Q16;
                    rda.rounds += 1;
                    rdb.rounds += 1;
                    let oa = a.offer(&pool[k][..end], 4, head, 16, &mut rda);
                    let ob = b.offer(&pool[k][..end], 4, head, 16, &mut rdb);
                    assert_eq!(oa, ob, "step {step}");
                    if let Some(o) = oa {
                        let acc = (rng.next() % 5) as usize;
                        let obs = [Obs { band: o.band, proposed: 4, accepted: acc.min(4), probing: o.probing }];
                        a.observe(&obs);
                        b.observe(&obs);
                    }
                }
            }
            assert_eq!(a.ids.digest(), b.ids.digest(), "step {step}");
            assert_eq!(a.rates, b.rates, "step {step}");
        }
        assert!(a.proposals + a.lookups > 0);
    }

    #[test]
    fn resolve_and_metrics() {
        assert_eq!(resolve(false, true), State::NotAsked);
        assert_eq!(resolve(true, false), State::OffNoCopySpan);
        assert_eq!(resolve(true, true), State::On);
        let mut o = String::new();
        metrics_append(&mut o);
        for name in ["velogb10_copy_span_record_state ", "velogb10_copy_span_record_ids_held ", "velogb10_copy_span_record_memory_bytes ",
                     "velogb10_copy_span_record_disk_bytes ", "velogb10_copy_span_record_proposals_total ",
                     "velogb10_copy_span_record_proposed_tokens_total ", "velogb10_copy_span_record_accepted_tokens_total ",
                     "velogb10_copy_span_record_landing_rate{band=\"64_up\"} ",
                     "velogb10_copy_span_record_fallback_total{reason=\"tp_digest\"} "] {
            assert!(o.contains(name), "missing {name}");
        }
        for f in Fallback::ALL { assert!(o.contains(&format!("reason=\"{}\"", f.reason()))); }
        assert!(state_json()["bands"].as_array().unwrap().len() == 3);
    }

    #[test]
    fn flags_are_cli_only_and_off_by_default() {
        let o = crate::opt!("copy-span-record").def();
        assert_eq!((o.def, o.env), ("off", ""));
        assert!(matches!(o.ty, crate::opts::Ty::Bool));
        assert_eq!(crate::opt!("copy-span-record-dir").def().env, "");
        assert_eq!(crate::opt!("copy-span-record-mib").def().def, DEFAULT_MIB.to_string());
        assert_eq!(crate::opt!("copy-span-record-mib").def().env, "");
        assert_eq!(crate::opt!("copy-span-record-read-only").def().def, "off");
        assert_eq!(crate::opt!("copy-span-record-read-only").def().env, "");
    }
}
