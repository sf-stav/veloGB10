//! Prometheus `/metrics` (community ask, plan CF-P1b): request-level serving metrics in the Prometheus text
//! exposition format, for both engines (NVFP4 and EXL3 share the HTTP layer) and every topology (the TP head
//! serves the API).
//!
//! Cost rule (owner 2026-10-02: on by default only if it costs nothing): nothing here runs on the GPU path or
//! inside the scheduler. A request carries one `Req` (a few plain fields, a local token counter); at request end
//! it does ~a dozen relaxed atomic adds. Everything else is computed when `/metrics` is scraped.
//!
//! Map of this directory (every item keeps its `crate::metrics::` path via `pub use self::<sub>::*`):
//! - `req.rs`: Request-level metrics: the fixed-bucket latency histograms, the per-finish and token totals, and the `Req` guard one request carries from admission to its end.
//! - `features.rs`: Engagement counters for request-visible features: tool-secret scrubbing, DF2 on the NVFP4 path, draft vocab, lane capacity, logprobs / schema / think-budget outcomes, over-cap admission.
//! - `exl3.rs`: EXL3 engagement and graceful-fallback counters: PLE / w4moe / fractional-width expert receipts, KV-cache and kernel-width fallbacks, forced reconstruct and width fallbacks.
//! - `sched.rs`: Long-run scheduler gauges and serving-path counters: rejects, lane-reserve admits, KV spill, prefill packing / interleave, naive-engine state, prefix pool, scheduler busy / age, graph entries.
//! - `fp4l.rs`: FP4-Local counters: host dispatches of the FP4-Local weight kernels by weight class and row class, the pack marker, the exact-precision twins and the resolved precision levers.
//! - `capture.rs`: CUDA graph captures that produced no graph: counters, the once-per-shape log line and what happens next per graph family.
//! - `toolcall.rs`: Tool-call parser shadow counting: how long the parser took, which rule was served, where the other rules would have differed, and the `render_toolcall` series.
//! - `lane_pack.rs`: Lane-pack and spec-lanes gauges: packed / serial rounds, skip reasons, in-flight and lane capacity, batched primes, the spec-lanes auto-policy estimates and meters.
//! - `render.rs`: The `/metrics` body: process gauges and the `render` function that writes every series in Prometheus text format 0.0.4.
//! - `tests.rs`: the unit tests (declared at the end of this file).
//!
//! The KV pool occupancy gauges (public issue #24) stay here with `render`'s callers.
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
use std::time::Instant;

mod req;
mod features;
mod exl3;
mod sched;
mod fp4l;
mod capture;
mod toolcall;
mod lane_pack;
mod render;

pub use self::req::*;
pub use self::features::*;
pub use self::exl3::*;
pub use self::sched::*;
pub use self::fp4l::*;
pub use self::capture::*;
pub use self::toolcall::*;
pub use self::lane_pack::*;
pub use self::render::*;

// ─── KV pool occupancy (public issue #24) ────────────────────────────────────────────────
// The KV cache is allocated once at boot: --max-batch lanes, each `window` positions long (the
// served --max-seq-len). These gauges say how much of that a server actually touches.
//
// The unit is a POSITION (one token of a lane's context), not a byte: a position costs the same
// bytes in every lane (`velogb10_kv_allocated_bytes` / `velogb10_kv_positions_capacity` is the per-position
// figure), and a lane's positions are a prefix 0..n of its own window; lanes do not share or borrow
// rows, so the per-lane series are what right-sizes --max-seq-len.
//
// Cost rule (as the rest of this file): the scheduler thread is the ONLY writer and publishes after
// each of its steps / prefill chunks with plain loads and stores (no RMW, no lock, nothing on the
// GPU path); everything else happens at scrape time. An engine that never calls
// `kv_pool_init` emits none of these series.

/// Per-lane series are emitted for at most this many lanes (label cardinality bound); the totals
/// always cover every lane.
pub const KV_LANE_SERIES_MAX: usize = 64;
/// Byte components of the boot allocation (`velogb10_kv_allocated_bytes{kind=...}`).
pub const KV_BYTE_KINDS: [&str; 3] = ["trunk_kv", "indexer_keys", "draft_head"];

struct KvLane {
    used: AtomicU64,
    cached: AtomicU64,
    peak: AtomicU64,
}

struct KvPool {
    window: u64,
    lanes: Box<[KvLane]>,
    used: AtomicU64,
    cached: AtomicU64,
    peak: AtomicU64,
}

static KV_POOL: std::sync::OnceLock<KvPool> = std::sync::OnceLock::new();
static KV_BYTES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static KV_BYTES_SET: AtomicU8 = AtomicU8::new(0);

fn kv_pool_new(lanes: usize, window: usize) -> KvPool {
    KvPool {
        window: window as u64,
        lanes: (0..lanes).map(|_| KvLane { used: AtomicU64::new(0), cached: AtomicU64::new(0), peak: AtomicU64::new(0) }).collect(),
        used: AtomicU64::new(0),
        cached: AtomicU64::new(0),
        peak: AtomicU64::new(0),
    }
}

/// Called once at boot by an engine that reports KV occupancy: `lanes` lanes of `window`
/// positions each (the served context window, after any clamp of --max-seq-len). First call wins.
pub fn kv_pool_init(lanes: usize, window: usize) {
    KV_POOL.get_or_init(|| kv_pool_new(lanes, window));
}

/// The bytes the boot allocation took for one component (`KV_BYTE_KINDS` index), on THIS process
/// (under TP: the head rank's share). Called by the model load; adds up over the layers.
pub fn kv_pool_add_bytes(kind: usize, bytes: u64) {
    if let Some(b) = KV_BYTES.get(kind) {
        b.fetch_add(bytes, Relaxed);
        KV_BYTES_SET.store(1, Relaxed);
    }
}

impl KvPool {
    /// One publication: per lane (positions a running request holds, positions an idle lane keeps
    /// for the prefix cache). Single writer (the scheduler thread). For a concurrent scrape: every
    /// peak is raised BEFORE the value it bounds, and that value is stored with Release and read
    /// with Acquire BEFORE its peak is read (`render`), so `peak >= used` holds in any scrape (two
    /// Relaxed stores to different atomics may become visible out of order on aarch64). Values are
    /// clamped to the lane window, so `used <= total` holds too. The per-lane series of one scrape
    /// can be one publication apart from the aggregate; they agree whenever the scheduler is not
    /// mid-publication.
    fn publish(&self, lanes: impl Iterator<Item = (usize, usize)>) {
        let (mut sum_used, mut sum_cached) = (0u64, 0u64);
        for (i, (used, cached)) in lanes.enumerate() {
            let Some(l) = self.lanes.get(i) else { break };
            let used = (used as u64).min(self.window);
            let cached = (cached as u64).min(self.window);
            sum_used += used;
            sum_cached += cached;
            if used > l.peak.load(Relaxed) { l.peak.store(used, Relaxed); }
            l.used.store(used, Release);
            l.cached.store(cached, Relaxed);
        }
        if sum_used > self.peak.load(Relaxed) { self.peak.store(sum_used, Relaxed); }
        self.used.store(sum_used, Release);
        self.cached.store(sum_cached, Relaxed);
    }

    fn render(&self, o: &mut String, bytes: Option<[u64; 3]>) {
        let n = self.lanes.len() as u64;
        let total = n * self.window;
        o.push_str("# HELP velogb10_kv_positions_capacity Static KV allocation in positions: --max-batch lanes x the served context window (--max-seq-len after any clamp). A position is one token of one lane's context, not a byte; every lane owns its own window and cannot borrow from another.\n# TYPE velogb10_kv_positions_capacity gauge\n");
        o.push_str(&format!("velogb10_kv_positions_capacity {total}\n"));
        o.push_str("# HELP velogb10_kv_positions_used Positions held by requests that are running now, summed over lanes: each lane's prompt (prefilled so far, including a prefix served from the cache) plus the tokens it has generated, to within one token (the newest token is stored when the next one is computed). Updated once per scheduler step and per prefill chunk. 0 when no request runs; speculative rows not yet accepted are not counted.\n# TYPE velogb10_kv_positions_used gauge\n");
        o.push_str(&format!("velogb10_kv_positions_used {}\n", self.used.load(Acquire))); // before the peak (see publish)
        o.push_str("# HELP velogb10_kv_positions_peak Highest velogb10_kv_positions_used since boot.\n# TYPE velogb10_kv_positions_peak gauge\n");
        o.push_str(&format!("velogb10_kv_positions_peak {}\n", self.peak.load(Relaxed)));
        o.push_str("# HELP velogb10_kv_positions_cached Prompt positions that idle lanes keep for the prefix cache, summed over lanes (reused when the next prompt on that lane starts with the same tokens, overwritten otherwise). Not part of positions_used; a lane that is running counts as used only.\n# TYPE velogb10_kv_positions_cached gauge\n");
        o.push_str(&format!("velogb10_kv_positions_cached {}\n", self.cached.load(Relaxed)));
        let shown = self.lanes.len().min(KV_LANE_SERIES_MAX);
        o.push_str("# HELP velogb10_kv_lane_positions_capacity One lane's window in positions (the same for every lane).\n# TYPE velogb10_kv_lane_positions_capacity gauge\n");
        for i in 0..shown { o.push_str(&format!("velogb10_kv_lane_positions_capacity{{lane=\"{i}\"}} {}\n", self.window)); }
        o.push_str("# HELP velogb10_kv_lane_positions_used Positions the request running on this lane holds now (0 = the lane is free). Sums to velogb10_kv_positions_used.\n# TYPE velogb10_kv_lane_positions_used gauge\n");
        for (i, l) in self.lanes.iter().take(shown).enumerate() {
            o.push_str(&format!("velogb10_kv_lane_positions_used{{lane=\"{i}\"}} {}\n", l.used.load(Acquire))); // before its peak
        }
        o.push_str("# HELP velogb10_kv_lane_positions_peak Highest velogb10_kv_lane_positions_used of this lane since boot (the longest context it has held; compare with velogb10_kv_lane_positions_capacity to size --max-seq-len).\n# TYPE velogb10_kv_lane_positions_peak gauge\n");
        for (i, l) in self.lanes.iter().take(shown).enumerate() {
            o.push_str(&format!("velogb10_kv_lane_positions_peak{{lane=\"{i}\"}} {}\n", l.peak.load(Relaxed)));
        }
        o.push_str("# HELP velogb10_kv_lane_positions_cached Prompt positions this lane keeps for the prefix cache while it is free (0 while a request runs on it).\n# TYPE velogb10_kv_lane_positions_cached gauge\n");
        for (i, l) in self.lanes.iter().take(shown).enumerate() {
            o.push_str(&format!("velogb10_kv_lane_positions_cached{{lane=\"{i}\"}} {}\n", l.cached.load(Relaxed)));
        }
        if let Some(b) = bytes {
            o.push_str("# HELP velogb10_kv_allocated_bytes Bytes allocated at boot for per-position state, on this process (with --tp: the head's share only). trunk_kv = the attention K/V cache in the --kv-cache format; indexer_keys = the sparse-attention index planes; draft_head = the speculation head's own cache. Fixed for the life of the process; the per-lane recurrent state and the prefix checkpoints are not included.\n# TYPE velogb10_kv_allocated_bytes gauge\n");
            for (k, name) in KV_BYTE_KINDS.iter().enumerate() {
                o.push_str(&format!("velogb10_kv_allocated_bytes{{kind=\"{name}\"}} {}\n", b[k]));
            }
        }
    }
}

/// The scheduler's publication (see `KvPool::publish`); a no-op on an engine without a pool.
#[inline]
pub fn kv_publish(lanes: impl Iterator<Item = (usize, usize)>) {
    if let Some(p) = KV_POOL.get() { p.publish(lanes); }
}

#[cfg(test)]
pub(crate) mod tests;
