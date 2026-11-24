//! `--dsv41-engram-iobench`: random record-read benchmark over the packed head files on this box.
//!
//! Cells: {io_uring O_DIRECT, thread-pool pread O_DIRECT, thread-pool pread buffered} x queue depth,
//! all closed loop (QD requests always in flight), each read = the aligned span covering one record.
//! Then the decode-step fetch: all rows one step needs for a rank's heads, issued at once.

use super::manifest::*;
use super::pack::parse_heads;
use super::uring::{AlignedBuf, Ring};
use anyhow::{anyhow, bail, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct HeadFile {
    ph: PackedHead,
    direct: std::fs::File,
    buffered: std::fs::File,
}

struct Pool {
    heads: Vec<HeadFile>,
    align: u64,
}

impl Pool {
    /// A random request: (head index in pool, aligned span).
    fn pick(&self, rng: &mut StdRng) -> (usize, ReadSpan) {
        let hi = rng.gen_range(0..self.heads.len());
        let m = &self.heads[hi].ph.manifest;
        let si = rng.gen_range(0..m.sections.len());
        let s = &m.sections[si];
        let row = rng.gen_range(0..s.rows);
        (hi, self.heads[hi].ph.span(s.layer_idx, s.g, row, self.align).unwrap())
    }
    fn pick_in(&self, hi: usize, si: usize, rng: &mut StdRng) -> ReadSpan {
        let s = &self.heads[hi].ph.manifest.sections[si];
        self.heads[hi].ph.span(s.layer_idx, s.g, rng.gen_range(0..s.rows), self.align).unwrap()
    }
    fn drop_cache(&self) {
        for h in &self.heads {
            unsafe { libc::posix_fadvise(h.buffered.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub ops: u64,
    pub errors: u64,
    pub secs: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}
impl Stats {
    fn from(mut lat_ns: Vec<u64>, errors: u64, secs: f64) -> Stats {
        lat_ns.sort_unstable();
        let q = |p: f64| lat_ns.get(((lat_ns.len() as f64 * p) as usize).min(lat_ns.len().saturating_sub(1))).copied().unwrap_or(0) as f64 / 1e3;
        Stats { ops: lat_ns.len() as u64, errors, secs, p50_us: q(0.5), p99_us: q(0.99), max_us: lat_ns.last().copied().unwrap_or(0) as f64 / 1e3 }
    }
    pub fn iops(&self) -> f64 {
        self.ops as f64 / self.secs
    }
}

fn touch(buf: &[u8], intra: usize) -> u8 {
    // read the record so the transfer cannot be optimised away
    buf[intra] ^ buf[intra + RECORD_BYTES - 1]
}

fn run_uring(pool: &Pool, qd: usize, secs: f64, seed: u64) -> Result<Stats> {
    let mut ring = Ring::new((qd as u32).next_power_of_two().max(2))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut bufs: Vec<AlignedBuf> = (0..qd).map(|_| AlignedBuf::new(8192)).collect();
    let mut start = vec![Instant::now(); qd];
    let mut want = vec![(0usize, 0usize); qd]; // (len, intra)
    let (mut lat, mut errors, mut sink) = (Vec::with_capacity(1 << 20), 0u64, 0u8);
    let t0 = Instant::now();
    let mut issue = |slot: usize, ring: &mut Ring, bufs: &mut Vec<AlignedBuf>, start: &mut Vec<Instant>, want: &mut Vec<(usize, usize)>, rng: &mut StdRng| {
        let (hi, sp) = pool.pick(rng);
        want[slot] = (sp.aligned_len, sp.intra);
        start[slot] = Instant::now();
        assert!(ring.push_read(pool.heads[hi].direct.as_raw_fd(), bufs[slot].as_mut_ptr(), sp.aligned_len as u32, sp.aligned_offset, slot as u64));
    };
    for s in 0..qd {
        issue(s, &mut ring, &mut bufs, &mut start, &mut want, &mut rng);
    }
    let mut inflight = qd;
    let mut stop = false;
    while inflight > 0 {
        ring.submit_and_wait(1)?;
        while let Some((ud, res)) = ring.pop() {
            let slot = ud as usize;
            let dt = start[slot].elapsed().as_nanos() as u64;
            inflight -= 1;
            if res as i64 != want[slot].0 as i64 {
                errors += 1;
            } else {
                sink ^= touch(bufs[slot].as_slice(), want[slot].1);
                if !stop {
                    lat.push(dt);
                }
            }
            if !stop && t0.elapsed().as_secs_f64() >= secs {
                stop = true;
            }
            if !stop {
                issue(slot, &mut ring, &mut bufs, &mut start, &mut want, &mut rng);
                inflight += 1;
            }
        }
    }
    std::hint::black_box(sink);
    Ok(Stats::from(lat, errors, t0.elapsed().as_secs_f64()))
}

fn run_threads(pool: &Pool, qd: usize, secs: f64, direct: bool, seed: u64) -> Result<Stats> {
    let stop = AtomicBool::new(false);
    let t0 = Instant::now();
    let res: Vec<(Vec<u64>, u64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..qd)
            .map(|t| {
                let stop = &stop;
                sc.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed + t as u64);
                    let mut buf = AlignedBuf::new(8192);
                    let (mut lat, mut errors, mut sink) = (Vec::with_capacity(1 << 16), 0u64, 0u8);
                    while !stop.load(Ordering::Relaxed) {
                        let (hi, sp) = pool.pick(&mut rng);
                        let f = if direct { &pool.heads[hi].direct } else { &pool.heads[hi].buffered };
                        let dst = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), sp.aligned_len) };
                        let t = Instant::now();
                        match f.read_exact_at(dst, sp.aligned_offset) {
                            Ok(()) => {
                                lat.push(t.elapsed().as_nanos() as u64);
                                sink ^= touch(buf.as_slice(), sp.intra);
                            }
                            Err(_) => errors += 1,
                        }
                    }
                    std::hint::black_box(sink);
                    (lat, errors)
                })
            })
            .collect();
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let secs = t0.elapsed().as_secs_f64();
    let errors = res.iter().map(|r| r.1).sum();
    Ok(Stats::from(res.into_iter().flat_map(|r| r.0).collect(), errors, secs))
}

/// One decode step: `tokens` x layers x group heads x ngram_cols rows, all issued at once through one ring.
fn run_step(pool: &Pool, group: &[usize], tokens: usize, reps: usize, seed: u64) -> Result<Stats> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_g = pool.heads[0].ph.manifest.geometry.ngram_cols;
    let n_layers = pool.heads[0].ph.manifest.geometry.layer_ids.len();
    let n = tokens * n_layers * group.len() * n_g;
    let mut ring = Ring::new((n as u32).next_power_of_two().max(2))?;
    let mut bufs: Vec<AlignedBuf> = (0..n).map(|_| AlignedBuf::new(8192)).collect();
    let mut want = vec![0usize; n];
    let (mut lat, mut errors) = (vec![], 0u64);
    for rep in 0..reps + 3 {
        let t = Instant::now();
        let mut k = 0;
        for _tok in 0..tokens {
            for &hi in group {
                for si in 0..n_layers * n_g {
                    let sp = pool.pick_in(hi, si, &mut rng);
                    want[k] = sp.aligned_len;
                    ring.push_read(pool.heads[hi].direct.as_raw_fd(), bufs[k].as_mut_ptr(), sp.aligned_len as u32, sp.aligned_offset, k as u64);
                    k += 1;
                }
            }
        }
        ring.submit_and_wait(n as u32)?;
        let mut got = 0;
        while got < n {
            while let Some((ud, res)) = ring.pop() {
                got += 1;
                if res as i64 != want[ud as usize] as i64 {
                    errors += 1;
                }
            }
            if got < n {
                ring.submit_and_wait(1)?;
            }
        }
        if rep >= 3 {
            lat.push(t.elapsed().as_nanos() as u64);
        }
    }
    let total: f64 = lat.iter().sum::<u64>() as f64 / 1e9;
    Ok(Stats::from(lat, errors, total))
}

fn row(label: &str, s: &Stats) -> String {
    format!("| {label} | {:.0} | {:.0} | {:.0} | {:.0} | {} |", s.iops(), s.p50_us, s.p99_us, s.max_us, s.errors)
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let arg = |k: &str| crate::arg_value(args, k).map(|s| s.to_string());
    let dir = PathBuf::from(arg("--engram-dir").ok_or_else(|| anyhow!("--engram-dir <pack dir> required"))?);
    let align: u64 = arg("--engram-iobench-align").and_then(|s| s.parse().ok()).unwrap_or(4096);
    if align != 512 && align != 4096 {
        bail!("--engram-iobench-align 512|4096");
    }
    let secs: f64 = arg("--engram-iobench-secs").and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let qds: Vec<usize> = arg("--engram-iobench-qds").unwrap_or_else(|| "1,8,32,128".into()).split(',').filter_map(|s| s.parse().ok()).collect();
    let modes = arg("--engram-iobench-modes").unwrap_or_else(|| "uring,threads,buffered".into());
    let steps = arg("--engram-iobench-steps").map(|s| s != "off").unwrap_or(true);
    let wanted = parse_heads(&arg("--engram-heads").unwrap_or_else(|| "0-7".into()), 8)?;
    let mut heads = vec![];
    for h in &wanted {
        if let Ok(ph) = PackedHead::open(&dir, *h) {
            let open = |direct: bool| {
                let mut o = std::fs::OpenOptions::new();
                o.read(true);
                if direct {
                    o.custom_flags(libc::O_DIRECT);
                }
                o.open(&ph.path)
            };
            let (direct, buffered) = (open(true)?, open(false)?);
            heads.push(HeadFile { ph, direct, buffered });
        }
    }
    if heads.is_empty() {
        bail!("no head packs found in {} for heads {wanted:?}", dir.display());
    }
    let layout = heads[0].ph.manifest.layout;
    let ids: Vec<usize> = heads.iter().map(|h| h.ph.manifest.head).collect();
    let total: u64 = heads.iter().map(|h| h.ph.manifest.file_bytes).sum();
    let pool = Pool { heads, align };
    println!("iobench: dir {} heads {ids:?} layout {} ({:.1} GB) align {align} secs/cell {secs}", dir.display(), layout.name(), total as f64 / 1e9);
    println!("\n| cell | IOPS | p50 us | p99 us | max us | errors |\n|---|---|---|---|---|---|");
    for qd in &qds {
        if modes.contains("uring") {
            let s = run_uring(&pool, *qd, secs, 1)?;
            println!("{}", row(&format!("io_uring O_DIRECT QD{qd}"), &s));
        }
    }
    for qd in &qds {
        if modes.contains("threads") {
            let s = run_threads(&pool, *qd, secs, true, 100)?;
            println!("{}", row(&format!("pread O_DIRECT {qd} threads"), &s));
        }
    }
    for qd in &qds {
        if modes.contains("buffered") {
            pool.drop_cache();
            let s = run_threads(&pool, *qd, secs, false, 200)?;
            println!("{}", row(&format!("pread buffered {qd} threads"), &s));
        }
    }
    if steps {
        println!("\nDecode-step fetch (io_uring O_DIRECT, all rows of the step issued at once; 6 rows/head/token = 2 layers x 3 columns)");
        println!("\n| rank heads | tokens | rows | p50 us | p99 us | max us | errors |\n|---|---|---|---|---|---|---|");
        for gs in [1usize, 2, 4, 8] {
            // aligned groups of gs heads whose files are all present on this box
            for base in (0..8).step_by(gs) {
                let idx: Vec<usize> = (base..base + gs).filter_map(|h| ids.iter().position(|x| *x == h)).collect();
                if idx.len() != gs {
                    continue;
                }
                for tokens in [1usize, 24] {
                    let s = run_step(&pool, &idx, tokens, 300, 7)?;
                    println!("| {:?} (TP{}) | {tokens} | {} | {:.0} | {:.0} | {:.0} | {} |", &ids[idx[0]..=idx[gs - 1]].iter().collect::<Vec<_>>(), 8 / gs, tokens * 6 * gs, s.p50_us, s.p99_us, s.max_us, s.errors);
                }
            }
        }
    }
    Ok(())
}
