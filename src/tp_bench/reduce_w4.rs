//! TP-4E: the world-4 prefill-sized fp32 all-reduce bench (`run_reduce_w4`) with its size / wire-byte helpers and the host
//! reference (the recursive-doubling tree at the epoch-aligned phase). See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------
// TP-4E: `--tp-reduce-bench --world 4` — the PREFILL-sized fp32 all-reduce at world > 2 on the REAL
// transport (no model), the same `crate::tp_xport` code the EXL3 engine launches. Additive: nothing above
// this line changed, and world 2 never reaches it (`run_tp_reduce_bench` dispatches on --world > 2).
//
// One process per rank, the same command on every box, `--peer-ips` rank-indexed. Per payload of the sweep
// (2 / 8 / 21 / 42 MB of fp32, whole 2560-float rows; 21 MB is the 2,048-row prefill chunk) it runs
// `--warmup` + `--reduces` reduces on transport `--xport 0|1|2` (0 serial single-block, 1 pipelined on one
// rail, 2 pipelined on both rails, rail 2 being a full N-way link), validates EVERY reduce bitwise against a
// host reference (the recursive-doubling tree at the R10-aligned phase), and reports per-reduce microseconds
// and GB/s per direction. Timing starts after a small serial all-reduce that re-aligns the ranks (host
// checks between reduces skew them), so a slow peer's host work is not billed to the transport.
//
// What it does NOT prove: a thermally/contention-loaded or model-coexisting rate (no model is resident), the
// in-model overlap hooks (no consumer), or serving (no sequence-parallel, no MoE). A throughput taken here is
// a transport number, not a prefill tok/s.
// ---------------------------------------------------------------------------------------------

pub struct ReduceW4Args {
    pub rank: i32,
    pub world: i32,
    /// rank-indexed RoCE/management IPs of all `world` ranks (both rails' TCP handshakes use them)
    pub peer_ips: Vec<String>,
    /// rail-1 control port; rail 2 takes `xtp::rail2_nway_base_for(world, port)`
    pub port: u16,
    pub dev: String,
    /// rail-2 device ("" = the engine default roceP2p1s0f1); used only with `xport == 2`
    pub dev2: String,
    pub gid: i32,
    /// 0 = serial single-block pairs, 1 = pipelined one rail, 2 = pipelined both rails
    pub xport: u8,
    pub sizes_mb: Vec<u64>,
    pub reduces: usize,
    pub warmup: usize,
    /// "f32" (in place) | "f32to16" (fp32 in, f16 out, the stage is the consumed src) |
    /// "f16" (f16 in/out in place, separate fp32 stage) | "f16w" (the same with the f16 first-round wire)
    pub io: String,
    pub lookahead: usize,
    pub blocks: u32,
    pub proxy_core: i32,
    pub proxy_core2: i32,
    pub main_core: i32,
}

/// Reduce sites per 2,048-row prefill chunk: 96 (48 layers x 2; PLAN/TP2_DESIGN_EXL3_2026-09-28.md section on prefill,
/// PLAN/notes_2026-09-28/TP-B.md, and the world-2 bench's own '96-reduce chunk' line). Used only to print an
/// order-of-magnitude extrapolation next to the per-reduce time; it is not a measured chunk time.
pub const W4_CHUNK_REDUCES: usize = 96;

/// The row width the payload sweep is rounded to (the Flash-Next hidden size, one fp32 row = 10,240 B).
pub const W4_ROW_FLOATS: usize = 2560;

/// fp32 floats of a `mb`-MB (decimal) payload, rounded DOWN to whole rows: 21 MB -> 2,050 rows, which is
/// the 2,048-row prefill chunk's 20 MiB (20.97 MB) within a row pair.
pub fn w4_sweep_floats(mb: u64) -> anyhow::Result<usize> {
    anyhow::ensure!((1..=512).contains(&mb), "sweep payload {mb} MB is outside 1..=512");
    let rows = (mb as usize * 1_000_000) / (W4_ROW_FLOATS * 4);
    anyhow::ensure!(rows >= 1, "sweep payload {mb} MB is under one row");
    Ok(rows * W4_ROW_FLOATS)
}

/// `--sizes-mb 2,8,21,42` -> [2, 8, 21, 42]; an empty or malformed list is an error, never a silent default.
pub fn w4_parse_sizes(s: &str) -> anyhow::Result<Vec<u64>> {
    let v: Vec<u64> = s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty())
        .map(|t| t.parse::<u64>().map_err(|e| anyhow::anyhow!("--sizes-mb entry '{t}': {e}"))).collect::<Result<_, _>>()?;
    anyhow::ensure!(!v.is_empty(), "--sizes-mb needs at least one payload size (MB)");
    for &m in &v { w4_sweep_floats(m)?; }
    Ok(v)
}

/// Bytes ONE rank sends per reduce on the wire (equal to what it receives): a recursive-doubling reduce runs
/// `rounds` exchanges of the whole payload; only the FIRST may be f16 (half the bytes), every later exchange
/// carries the fp32 running sum (`tp_xport::Io` docs). World 2 is the one-round case.
pub fn w4_wire_bytes(n_floats: usize, rounds: usize, wire_f16: bool) -> u64 {
    let f32b = n_floats as u64 * 4;
    let first = if wire_f16 { f32b / 2 } else { f32b };
    first + f32b * (rounds.saturating_sub(1)) as u64
}

/// The combinations the bench accepts. Transport 0 is the in-place fp32 serial pair (the serving path
/// converts around it), so any f16 IO there would be a different experiment, refused rather than silently
/// upgraded to the pipelined transport.
pub fn w4_check_combo(xport: u8, io: &str) -> anyhow::Result<usize> {
    anyhow::ensure!(xport <= 2, "--xport {xport}: 0 (serial) | 1 (pipelined, one rail) | 2 (pipelined, both rails)");
    anyhow::ensure!(matches!(io, "f32" | "f32to16" | "f16" | "f16w"), "--io '{io}': f32 | f32to16 | f16 | f16w");
    anyhow::ensure!(xport > 0 || io == "f32",
        "--xport 0 (serial K1/K2 pairs) carries only --io f32 in place; '{io}' needs the pipelined transport (--xport 1 or 2)");
    Ok(if xport == 2 { 2 } else { 1 })
}

/// The recursive-doubling sums of one element across `world` ranks, every rank's final value. Exchange k pairs
/// ranks across bit `(k + shift) % rounds`; `shift` is the R10 phase: the engine aligns both rails to an even
/// device epoch, so the first exchange of a chunk runs round 1 (partner ^ 2) and the world-4 result is
/// (p0+p2)+(p1+p3), i.e. `shift` 1. fp32 addition is commutative, so the two operands' order is immaterial.
pub(super) fn w4_tree_all(vals: &[f32; 8], world: usize, shift: usize) -> [f32; 8] {
    let rounds = world.trailing_zeros() as usize;
    let mut v = *vals;
    for k in 0..rounds {
        let bit = 1usize << ((k + shift) % rounds);
        let mut nv = v;
        for r in 0..world { nv[r] = v[r & !bit] + v[r | bit]; }
        v = nv;
    }
    v
}

/// The W=4 bench fill: `fill_val` everywhere except every 16th element, which is a CANCELLATION quad that makes the
/// two round-phase associations differ in EVERY io mode: rank 0 = +2048, rank 2 = -2048 and ranks 1 / 3 = tiny f16-exact
/// values (2^-14 (1 + k/1024)), far below half an fp32 ulp of 2048. At the aligned phase ((p0+p2)+(p1+p3)) the big pair
/// cancels exactly and the result is t1+t3 (about 2^-13); at the other phase ((p0+p1)+(p2+p3)) both tiny values are
/// absorbed and the result is 0. Without it, f16-exact partials of similar magnitude sum exactly in fp32 in either order
/// and the f16 modes could not detect a phase shift at all (found by `checker_detects_a_flipped_bit_and_a_phase_shift`).
/// Only valid for world 4 (ranks 0..3); other ranks fall back to `fill_val`.
pub fn w4_fill(r: i32, i: usize, it: usize) -> f32 {
    if i % 16 != 0 || !(0..4).contains(&r) { return fill_val(r, i, it); }
    let k = (((i / 16) + it) * 7 % 1024) as f32;
    let k3 = ((((i / 16) + it) * 7 + 512) % 1024) as f32;
    let tiny = |k: f32| (1.0 + k / 1024.0) * 2f32.powi(-14);
    match r { 0 => 2048.0, 2 => -2048.0, 1 => tiny(k), _ => tiny(k3) }
}

/// The expected output bits of element `i` of reduce `it` for `io`: fp32 bits for "f32", f16 bits for the
/// three f16-out IOs. "f16" / "f16w" start from f16-exact partials (the rank's fill rounded to f16 first).
pub fn w4_expected_bits(io: &str, world: usize, i: usize, it: usize, shift: usize) -> u32 {
    let mut v = [0f32; 8];
    for r in 0..world {
        let x = w4_fill(r as i32, i, it);
        v[r] = if matches!(io, "f16" | "f16w") { half::f16::from_f32(x).to_f32() } else { x };
    }
    let s = w4_tree_all(&v, world, shift)[0];
    if io == "f32" { s.to_bits() } else { half::f16::from_f32(s).to_bits() as u32 }
}

/// Count the elements of `got` (output bits, see `w4_expected_bits`) that differ from the host reference,
/// and return the first mismatch (index, got, want). Multithreaded: a 42 MB payload is 10.5 M elements x
/// `world` hashes.
pub fn w4_count_bad(io: &str, world: usize, it: usize, shift: usize, n: usize,
                    got: &(dyn Fn(usize) -> u32 + Sync)) -> (usize, Option<(usize, u32, u32)>) {
    let threads = 8usize.min(n.max(1));
    let per = n.div_ceil(threads);
    let parts: Vec<(usize, Option<(usize, u32, u32)>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| sc.spawn(move || {
            let (lo, hi) = (t * per, ((t + 1) * per).min(n));
            let (mut bad, mut first) = (0usize, None);
            for i in lo..hi {
                let (g, w) = (got(i), w4_expected_bits(io, world, i, it, shift));
                if g != w { bad += 1; if first.is_none() { first = Some((i, g, w)); } }
            }
            (bad, first)
        })).collect();
        hs.into_iter().map(|h| h.join().expect("w4 checker thread")).collect()
    });
    let bad = parts.iter().map(|p| p.0).sum();
    (bad, parts.iter().filter_map(|p| p.1).next())
}

pub fn run_reduce_w4(a: ReduceW4Args) -> anyhow::Result<()> {
    use crate::exl3_forward::xtp;
    let nr = w4_check_combo(a.xport, &a.io)?;
    anyhow::ensure!(a.world > 2 && (a.world as u32).is_power_of_two(), "--world {} (this harness is world > 2, a power of two; world 2 is the TP-F bench)", a.world);
    anyhow::ensure!(a.rank >= 0 && a.rank < a.world, "--rank {} outside 0..{}", a.rank, a.world);
    anyhow::ensure!(a.peer_ips.len() >= a.world as usize, "--peer-ips needs {} rank-indexed addresses (got {})", a.world, a.peer_ips.len());
    anyhow::ensure!(a.reduces >= 1, "--reduces must be >= 1");
    let rounds = crate::tp_xport::rounds_of(a.world as usize)?;
    let shift = 1 % rounds;   // R10: an even aligned start runs round (0 + 1) % rounds first
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..a.world as usize].iter().map(|s| s.parse())
        .collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!("--peer-ips: {e}"))?;
    let dev2 = if a.dev2.is_empty() { "roceP2p1s0f1".to_string() } else { a.dev2.clone() };
    println!("=== --tp-reduce-bench --world {} rank {} xport {} ({} rail(s)) io {} sizes {:?} MB reduces {} (+{} warmup) lookahead {} blocks {} ===",
             a.world, a.rank, a.xport, nr, a.io, a.sizes_mb, a.reduces, a.warmup, a.lookahead, a.blocks);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {}", a.main_core);
    }
    // rail 1: the N-way link, exactly as tp::bring_up builds it; the proxy owns it from here
    let mut link = TpLink::connect_nway(a.rank, a.world, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(W4_ROW_FLOATS * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    let mut rails = vec![ctx];
    let mut link2: Option<std::mem::ManuallyDrop<TpLink>> = None;
    if nr == 2 {
        // a box whose second device is unusable must fail by name here: its three peers are about to block in accept()
        xtp::rail_local_preflight(&dev2, a.gid)?;
        let base2 = xtp::rail2_nway_base_for(a.world, a.port)?;
        let mut l2 = TpLink::connect_nway(a.rank, a.world, &ips, base2, &dev2, a.gid, crate::tp::TP_SLOT_BYTES)?;
        l2.set_payload(W4_ROW_FLOATS * 4, true)?;
        rails.push(l2.ctx_device_ptr());
        let l2 = std::mem::ManuallyDrop::new(l2);
        net::spawn_proxy_aux(l2.ctx_addr(), a.proxy_core2);
        println!("[w4] rank {} rail 2 UP: {dev2} (N-way, TCP base {base2}), proxy core {}", a.rank, a.proxy_core2);
        link2 = Some(l2);
    }
    std::thread::sleep(std::time::Duration::from_millis(300));
    // a fresh process: both rails start at epoch 0. A nonzero, non-multiple start would change the association
    // the reference assumes, so it is refused instead of "adjusted" (the engine pads; a bench has no reason to).
    let e1 = net::ctx_device_epoch(link.ctx_addr());
    let e2 = link2.as_ref().map(|l| net::ctx_device_epoch(l.ctx_addr()));
    anyhow::ensure!(e1 % rounds as u64 == 0 && e2.map_or(true, |e| e % rounds as u64 == 0),
                    "rail epochs {e1}/{e2:?} are not multiples of {rounds} on a fresh link");
    let pipe = if a.xport > 0 { Some(crate::tp_xport::Pipe::new(&dev, rails, a.world as usize, a.lookahead, a.blocks)?) } else { None };
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream
    let mut sync_buf = dev.alloc_zeros::<f32>(W4_ROW_FLOATS)?;
    dev.memset_zeros(&mut sync_buf)?;   // AGENTS §2.2: alloc_zeros does not zero
    dev.synchronize()?;
    let sync_p = *sync_buf.device_ptr() as u64;
    let wire_f16 = a.io == "f16w";
    let (mut total_bad, mut total_reduces) = (0usize, 0usize);
    let mut rows_out: Vec<String> = Vec::new();
    for &mb in &a.sizes_mb {
        let n = w4_sweep_floats(mb)?;
        anyhow::ensure!(crate::tp_xport::pipe_ok(n), "payload {n} floats is not 16 B-aligned for the pipelined transport");
        let mut buf32 = dev.alloc_zeros::<f32>(if a.io == "f16" || a.io == "f16w" { 4 } else { n })?;
        let mut out16 = dev.alloc_zeros::<u16>(if a.io == "f32" { 4 } else { n })?;
        dev.memset_zeros(&mut out16)?;   // AGENTS §2.2: alloc_zeros does not zero
        dev.synchronize()?;
        let mid32 = dev.alloc_zeros::<f32>(if a.io == "f16" || a.io == "f16w" { n } else { 4 })?;
        let mut host32 = vec![0f32; if a.io == "f32" || a.io == "f32to16" { n } else { 0 }];
        let mut host16 = vec![0u16; if a.io == "f16" || a.io == "f16w" { n } else { 0 }];
        let (mut times, mut bad_here) = (Vec::with_capacity(a.reduces), 0usize);
        for it in 0..a.warmup + a.reduces {
            // fill this rank's partial (uploaded and synced BEFORE the timed region)
            match a.io.as_str() {
                "f16" | "f16w" => {
                    for (i, v) in host16.iter_mut().enumerate() { *v = half::f16::from_f32(w4_fill(a.rank, i, it)).to_bits(); }
                    dev.htod_sync_copy_into(&host16, &mut out16)?;
                }
                _ => {
                    for (i, v) in host32.iter_mut().enumerate() { *v = w4_fill(a.rank, i, it); }
                    dev.htod_sync_copy_into(&host32, &mut buf32)?;
                }
            }
            dev.synchronize()?;
            // re-align the ranks: a small serial all-reduce on rail 1 (rounds epochs, phase preserved)
            crate::tp_xport::reduce_serial(&kern, &stream, ctx, sync_p, W4_ROW_FLOATS, rounds)?;
            dev.synchronize()?;
            let (p32, p16, pm) = (*buf32.device_ptr() as u64, *out16.device_ptr() as u64, *mid32.device_ptr() as u64);
            let t0 = std::time::Instant::now();
            match (&pipe, a.io.as_str()) {
                (None, _) => crate::tp_xport::reduce_serial(&kern, &stream, ctx, p32, n, rounds)?,
                (Some(x), "f32") => crate::tp_xport::reduce_pipe(&kern, &stream, x, crate::tp_xport::Io::f32_inplace(p32), n)?,
                (Some(x), "f32to16") => crate::tp_xport::reduce_pipe(&kern, &stream, x,
                    crate::tp_xport::Io { src: p32, src_f16: false, out: p16, out_f16: true, wire_f16: false, mid: p32 }, n)?,
                (Some(x), _) => crate::tp_xport::reduce_pipe(&kern, &stream, x,
                    crate::tp_xport::Io { src: p16, src_f16: true, out: p16, out_f16: true, wire_f16, mid: pm }, n)?,
            }
            dev.synchronize()?;
            let dt = t0.elapsed().as_nanos() as u64;
            if it >= a.warmup { times.push(dt); }
            // validate every reduce, warmup included
            let (bad, first, alt) = if a.io == "f32" {
                let got = dev.dtoh_sync_copy(&buf32)?;
                let g = |i: usize| got[i].to_bits();
                let (b, f) = w4_count_bad(&a.io, a.world as usize, it, shift, n, &g);
                let alt = if b > 0 { w4_count_bad(&a.io, a.world as usize, it, (shift + 1) % rounds, n, &g).0 } else { usize::MAX };
                (b, f, alt)
            } else {
                let got = dev.dtoh_sync_copy(&out16)?;
                let g = |i: usize| got[i] as u32;
                let (b, f) = w4_count_bad(&a.io, a.world as usize, it, shift, n, &g);
                let alt = if b > 0 { w4_count_bad(&a.io, a.world as usize, it, (shift + 1) % rounds, n, &g).0 } else { usize::MAX };
                (b, f, alt)
            };
            if bad > 0 {
                if let Some((i, g, w)) = first { println!("  MISMATCH {mb} MB reduce {it} elem {i}: got {g:#010x} want {w:#010x} ({bad} of {n} differ)"); }
                if alt == 0 { println!("  NOTE: the output equals the OTHER round-phase association exactly (an R10 phase shift, not corruption)"); }
            }
            bad_here += bad;
            total_reduces += 1;
            if link.abort_status() != 0 { anyhow::bail!("rail-1 transport ABORT status {} at {mb} MB reduce {it}", link.abort_status()); }
            if let Some(l2) = &link2 { if l2.abort_status() != 0 { anyhow::bail!("rail-2 ABORT status {} at {mb} MB reduce {it}", l2.abort_status()); } }
        }
        total_bad += bad_here;
        let mut s = times.clone();
        s.sort_unstable();
        let p50_us = pct(&s, 0.5) as f64 / 1e3;
        let wire = w4_wire_bytes(n, rounds, wire_f16);
        let algo_gbs = n as f64 * 4.0 / (p50_us * 1e3);
        let wire_gbs = wire as f64 / (p50_us * 1e3);
        println!("  {mb:>3} MB ({:.2} MiB, {} rows) xport {} io {}: p10 {:.0} p50 {:.0} p90 {:.0} max {:.0} us | payload {:.2} GB/s | wire {:.2} GB/s per direction \
                  ({:.2} per rail, {} rail(s), {:.1} MB sent/reduce) | x{} reduce sites = {:.1} ms per chunk (extrapolation) | {}",
                 n as f64 * 4.0 / 1048576.0, n / W4_ROW_FLOATS, a.xport, a.io, pct(&s, 0.1) as f64 / 1e3, p50_us, pct(&s, 0.9) as f64 / 1e3,
                 pct(&s, 1.0) as f64 / 1e3, algo_gbs, wire_gbs, wire_gbs / nr as f64, nr, wire as f64 / 1e6, W4_CHUNK_REDUCES, p50_us * W4_CHUNK_REDUCES as f64 / 1e3,
                 if bad_here == 0 { "BITWISE_OK" } else { "FAIL" });
        rows_out.push(format!("W4SWEEP rank {} world {} xport {} io {} mb {} p50_us {:.0} payload_gbs {:.2} wire_gbs_per_dir {:.2} ms_x{} {:.1} bitwise {}",
                              a.rank, a.world, a.xport, a.io, mb, p50_us, algo_gbs, wire_gbs, W4_CHUNK_REDUCES, p50_us * W4_CHUNK_REDUCES as f64 / 1e3,
                              if bad_here == 0 { "OK" } else { "FAIL" }));
    }
    let (f1, g1) = (net::ctx_device_epoch(link.ctx_addr()), link.gpu_ready());
    println!("  rail 1: tail fires {}  gate binds {}  abort {}  device epoch {f1} gpu_ready {g1}", link.tail_fires(), link.gate_waits(), link.abort_status());
    let mut phase_ok = f1 % rounds as u64 == 0;
    if let Some(l2) = &link2 {
        let f2 = net::ctx_device_epoch(l2.ctx_addr());
        println!("  rail 2: tail fires {}  abort {}  device epoch {f2} gpu_ready {}", l2.tail_fires(), l2.abort_status(), l2.gpu_ready());
        phase_ok &= f2 % rounds as u64 == 0;
    }
    println!("  round phase after the run: {} (every rail's device epoch must stay a multiple of {rounds})", if phase_ok { "INTACT" } else { "DRIFTED" });
    for r in &rows_out { println!("{r}"); }
    println!("  VALIDATION: {total_bad} mismatching elements over {total_reduces} reduces => {}",
             if total_bad == 0 && phase_ok { "BITWISE_OK" } else { "FAIL" });
    if total_bad != 0 || !phase_ok { anyhow::bail!("world-{} reduce bench validation FAILED", a.world); }
    Ok(())
}
