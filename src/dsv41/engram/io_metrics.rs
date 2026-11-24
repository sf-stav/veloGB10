//! Engram runtime counters (leg EIO) for `/metrics`: process-wide relaxed atomics that the runtime
//! (`prefetch::EngramIo`) bumps and `render` prints. Nothing is printed until a runtime has opened on this
//! process (`ACTIVE`), so every other family's `/metrics` body stays byte-identical.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

pub static ACTIVE: AtomicBool = AtomicBool::new(false);
/// per kind (0 = decode rounds, 1 = prefill chunks): submitted; published counts per-layer decode steps
pub static STEPS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static PUBLISHED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// rows asked for by steps, by outcome
pub static ROWS_HIT: AtomicU64 = AtomicU64::new(0);
pub static ROWS_MISS: AtomicU64 = AtomicU64::new(0);
pub static ROWS_DEAD: AtomicU64 = AtomicU64::new(0);
/// warm requests (draft-token prefetch, prompt read-ahead): rows already resident / rows read
pub static WARM_RESIDENT: AtomicU64 = AtomicU64::new(0);
pub static WARM_READS: AtomicU64 = AtomicU64::new(0);
pub static EVICTIONS: AtomicU64 = AtomicU64::new(0);
pub static READS: AtomicU64 = AtomicU64::new(0);
pub static READ_BYTES: AtomicU64 = AtomicU64::new(0);
pub static READ_RETRIES: AtomicU64 = AtomicU64::new(0);
pub static READ_ERRORS: AtomicU64 = AtomicU64::new(0);
/// reads slower than `io::SLOW_READ` (500 us) and the longest read (us)
pub static SLOW_READS: AtomicU64 = AtomicU64::new(0);
pub static MAX_READ_US: AtomicU64 = AtomicU64::new(0);
/// the planner waited for the GPU to free a descriptor-ring slot / for a cache slot to retire
pub static RING_WAITS: AtomicU64 = AtomicU64::new(0);
pub static CACHE_FULL_WAITS: AtomicU64 = AtomicU64::new(0);
/// GPU side (from the channels' control blocks): steps whose rows were late, ns waited, bounded-wait timeouts
pub static STALLS: AtomicU64 = AtomicU64::new(0);
pub static STALL_NS: AtomicU64 = AtomicU64::new(0);
pub static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
pub static CACHE_SLOTS: AtomicU64 = AtomicU64::new(0);
pub static CACHE_OCCUPIED: AtomicU64 = AtomicU64::new(0);
pub static READERS: AtomicU64 = AtomicU64::new(0);
/// leg HT1: the readers' idle tier (park entries, ns parked, wake-ups back to the awake tier), live (a long park adds every 200 ms)
pub static READER_PARKS: AtomicU64 = AtomicU64::new(0);
pub static READER_PARKED_NS: AtomicU64 = AtomicU64::new(0);
pub static READER_WAKES: AtomicU64 = AtomicU64::new(0);

/// Host submit -> publish latency of decode steps, microsecond bucket upper bounds (+Inf last).
pub const LAT_BOUNDS_US: [u64; 10] = [50, 100, 200, 300, 500, 750, 1000, 2000, 5000, 20000];
pub static LAT_BUCKETS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
pub static LAT_SUM_US: AtomicU64 = AtomicU64::new(0);

pub fn observe_decode_latency_us(us: u64) {
    let i = LAT_BOUNDS_US.iter().position(|&b| us <= b).unwrap_or(LAT_BOUNDS_US.len());
    LAT_BUCKETS[i].fetch_add(1, Relaxed);
    LAT_SUM_US.fetch_add(us, Relaxed);
}

fn line(o: &mut String, name: &str, kind: &str, help: &str, v: u64) {
    o.push_str(&format!("# HELP velogb10_dsv41_engram_{name} {help}\n# TYPE velogb10_dsv41_engram_{name} {kind}\nvelogb10_dsv41_engram_{name} {v}\n"));
}

/// The `velogb10_dsv41_engram_*` series (nothing when no Engram runtime opened on this process).
pub fn render(o: &mut String) {
    if ACTIVE.load(Relaxed) { render_body(o); }
}

fn render_body(o: &mut String) {
    let g = |a: &AtomicU64| a.load(Relaxed);
    o.push_str("# HELP velogb10_dsv41_engram_steps_total DeepSeek-V4.1 Engram (leg EIO): row steps submitted, by channel (decode = verify rounds, prefill = prompt chunks).\n# TYPE velogb10_dsv41_engram_steps_total counter\n");
    o.push_str(&format!("velogb10_dsv41_engram_steps_total{{channel=\"decode\"}} {}\nvelogb10_dsv41_engram_steps_total{{channel=\"prefill\"}} {}\n", g(&STEPS[0]), g(&STEPS[1])));
    o.push_str("# HELP velogb10_dsv41_engram_rows_total Engram rows asked for by steps: hit = in the pinned row cache, miss = read from the NVMe, dead = image positions (gate 0, nothing read).\n# TYPE velogb10_dsv41_engram_rows_total counter\n");
    o.push_str(&format!("velogb10_dsv41_engram_rows_total{{outcome=\"hit\"}} {}\nvelogb10_dsv41_engram_rows_total{{outcome=\"miss\"}} {}\nvelogb10_dsv41_engram_rows_total{{outcome=\"dead\"}} {}\n",
                        g(&ROWS_HIT), g(&ROWS_MISS), g(&ROWS_DEAD)));
    o.push_str("# HELP velogb10_dsv41_engram_warm_rows_total Rows of warm requests (draft-token prefetch, prompt read-ahead): resident = already cached, read = read ahead of their step.\n# TYPE velogb10_dsv41_engram_warm_rows_total counter\n");
    o.push_str(&format!("velogb10_dsv41_engram_warm_rows_total{{outcome=\"resident\"}} {}\nvelogb10_dsv41_engram_warm_rows_total{{outcome=\"read\"}} {}\n", g(&WARM_RESIDENT), g(&WARM_READS)));
    line(o, "stalls_total", "counter", "Steps whose rows were not published when the GPU reached them (the in-stream wait had to wait). 0 = Engram never delayed a round.", g(&STALLS));
    line(o, "stall_microseconds_total", "counter", "Microseconds the GPU waited for late Engram rows.", g(&STALL_NS) / 1000);
    o.push_str("# HELP velogb10_dsv41_engram_published_total Engram steps published to the GPU (decode: one per Engram layer per round).\n# TYPE velogb10_dsv41_engram_published_total counter\n");
    o.push_str(&format!("velogb10_dsv41_engram_published_total{{channel=\"decode\"}} {}\nvelogb10_dsv41_engram_published_total{{channel=\"prefill\"}} {}\n", g(&PUBLISHED[0]), g(&PUBLISHED[1])));
    line(o, "timeouts_total", "counter", "Bounded waits that timed out (the round failed; never served from unwritten rows).", g(&TIMEOUTS));
    line(o, "reads_total", "counter", "Record reads completed by the reader pool (O_DIRECT, io_uring).", g(&READS));
    line(o, "read_bytes_total", "counter", "Bytes the reads transferred (aligned spans).", g(&READ_BYTES));
    line(o, "read_retries_total", "counter", "Reads retried after EINTR / EAGAIN.", g(&READ_RETRIES));
    line(o, "slow_reads_total", "counter", "Reads that took longer than 500 us from submit to completion (the NVMe tail behind late rows).", g(&SLOW_READS));
    line(o, "max_read_microseconds", "gauge", "The longest single read since boot, us.", g(&MAX_READ_US));
    line(o, "read_errors_total", "counter", "Reads that failed (media error or short read): their steps failed.", g(&READ_ERRORS));
    line(o, "evictions_total", "counter", "Row-cache slots recycled (CLOCK).", g(&EVICTIONS));
    line(o, "ring_waits_total", "counter", "Submits that waited for the GPU to free a descriptor-ring slot.", g(&RING_WAITS));
    line(o, "cache_full_waits_total", "counter", "Submits that waited because every cache slot was held by an unretired step.", g(&CACHE_FULL_WAITS));
    line(o, "cache_slots", "gauge", "Row-cache slots (--engram-row-cache-mib / 272 B).", g(&CACHE_SLOTS));
    line(o, "cache_occupied_slots", "gauge", "Row-cache slots holding a row.", g(&CACHE_OCCUPIED));
    line(o, "readers", "gauge", "Reader threads (--engram-readers).", g(&READERS));
    line(o, "reader_parks_total", "counter", "Leg HT1: times a reader parked (slept until work) after --engram-park-us without a read; about once per quiet gap per reader, never once per round.", g(&READER_PARKS));
    o.push_str(&format!("# HELP velogb10_dsv41_engram_reader_parked_seconds_total Leg HT1: seconds the readers spent parked instead of spinning a core (summed over readers).\n# TYPE velogb10_dsv41_engram_reader_parked_seconds_total counter\nvelogb10_dsv41_engram_reader_parked_seconds_total {:.3}\n", g(&READER_PARKED_NS) as f64 / 1e9));
    line(o, "reader_wakes_total", "counter", "Leg HT1: parked readers woken back to the spinning tier by work or by a request starting.", g(&READER_WAKES));
    o.push_str("# HELP velogb10_dsv41_engram_decode_publish_microseconds Decode rounds, first Engram layer (the critical rows): host submit (tokens known) to rows published to the GPU.\n# TYPE velogb10_dsv41_engram_decode_publish_microseconds histogram\n");
    let mut acc = 0u64;
    for (i, b) in LAT_BOUNDS_US.iter().enumerate() {
        acc += g(&LAT_BUCKETS[i]);
        o.push_str(&format!("velogb10_dsv41_engram_decode_publish_microseconds_bucket{{le=\"{b}\"}} {acc}\n"));
    }
    acc += g(&LAT_BUCKETS[LAT_BOUNDS_US.len()]);
    o.push_str(&format!("velogb10_dsv41_engram_decode_publish_microseconds_bucket{{le=\"+Inf\"}} {acc}\nvelogb10_dsv41_engram_decode_publish_microseconds_sum {}\nvelogb10_dsv41_engram_decode_publish_microseconds_count {acc}\n", g(&LAT_SUM_US)));
}

#[cfg(test)]
mod tests {
    #[test]
    fn silent_until_active_then_prometheus_text() {
        let mut o = String::new();
        // no runtime opens in the unit tests: the shared /metrics body gets nothing from this file
        super::render(&mut o);
        assert!(o.is_empty());
        super::observe_decode_latency_us(120);
        super::render_body(&mut o);
        assert!(o.contains("velogb10_dsv41_engram_stalls_total "));
        assert!(o.contains("velogb10_dsv41_engram_reader_parked_seconds_total ") && o.contains("velogb10_dsv41_engram_reader_wakes_total "));
        assert!(o.contains("velogb10_dsv41_engram_decode_publish_microseconds_bucket{le=\"+Inf\"}"));
        for l in o.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "bad sample line {l}");
        }
    }
}
