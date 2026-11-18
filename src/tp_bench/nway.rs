//! TP-4C benches over the N-way link: the decode all-reduce at world 2 / 4 (`run_reduce_dec_nway`), the single-rail variant
//! and the world-4 vocab-parallel gather. They launch exactly the engine's `tp_xport` sequence. See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------
// TP-4C: `--tp-reduce-bench --mode decw|decwf32|decwserial|decwkeys|decwmix` — the DECODE-path
// all-reduce at world 2 or 4 (rounds = log2(world) exchanges per logical reduce) on the REAL
// transport, no model. A SEPARATE function over `connect_nway` (the TP-G `run_reduce_dec` above is
// untouched and stays the world-2 pair bench). It launches exactly the sequence the engine launches
// (`tp_xport::dec_exchanges` / `dec_exchange_args` / `dec_intermediate`, the same kernels) and checks
// every result BITWISE against the host reference (`tp_xport::rd_tree_reference`: the recursive-doubling
// association the epoch phase fixes; keys: the max of the ranks' keys). The device epoch counter is read
// back after every timed batch and must have advanced by EXACTLY `rounds` per logical reduce — the
// hardware proof of the R10 epoch-phase discipline that the CPU model in tp_xport.rs only assumes.
//   decw       : producer with folded K1 (xq_ks_combine_f32_k1l) -> rounds x K2dec, fp32 intermediate in the
//                partial buffer, f16 out on the last round   (the mixer out-proj / MoE combine arm)
//   decwf32    : the same with an fp32 `out` that differs from the zero-free `local` (the draft-head screen
//                arm: the intermediate is `out`, `local` must survive unchanged)
//   decwserial : producer + reduce_serial(rounds) + f16 cvt (the TP-B..F serial sequence; the A/B for decw)
//   decwkeys   : vocab-parallel key merge — keys K1 (plain, every round) + xq_tp_wait_keys per round,
//                merged ids vs the host max (ties broken to the lowest id, the total order)
//   decwmix    : all four arms interleaved in a rotating order, every reduce a different shape: any arm
//                consuming a wrong number of epochs flips the round association of a LATER arm and fails
// ---------------------------------------------------------------------------------------------

pub struct DecWorldArgs {
    pub rank: i32,
    pub world: u32,
    /// peer IPs indexed by RANK (comma list on the CLI; entry `[rank]` is this node's own, unused)
    pub peer_ips: Vec<String>,
    pub port: u16,
    pub dev: String,
    pub gid: i32,
    pub floats: usize,
    /// timed batches (each 96 logical reduces per arm; the first batch is validated but excluded from the percentiles)
    pub reduces: usize,
    pub mode: String,
    pub blocks: u32,
    pub proxy_core: i32,
    pub main_core: i32,
}

/// A vocab-parallel argmax key (value in the high 32 bits, `0xFFFFFFFF - global_id` in the low 32): only a few
/// distinct values, so equal-value ties across ranks are frequent and the lowest-id tie-break is exercised.
pub(super) fn dec_key(rank: i32, row: usize, it: usize) -> u64 {
    let mut z = (((it as u64) << 32) | row as u64).wrapping_add((rank as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let value = 1 + (z >> 32) % 5;
    let id = rank as u64 * 62_080 + (z & 0xFFFF_FFFF) % 62_080;
    (value << 32) | (0xFFFF_FFFFu64 - id)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecArm { Dec16, Dec32, Serial, Keys }

pub fn run_reduce_dec_nway(a: DecWorldArgs) -> anyhow::Result<()> {
    if a.mode == "decwvpg" { return run_vp_gather_nway(a); }   // TP-4H2: the vocab-parallel row all-gather arm
    if a.mode == "decws" { return run_reduce_single_nway(a); } // TP-4F2: the single-stage vs rd decode reduce A/B
    const NB: usize = 96;
    const MKEYS: usize = 16;
    let active: Vec<DecArm> = match a.mode.as_str() {
        "decw" => vec![DecArm::Dec16],
        "decwf32" => vec![DecArm::Dec32],
        "decwserial" => vec![DecArm::Serial],
        "decwkeys" => vec![DecArm::Keys],
        "decwmix" => vec![DecArm::Dec16, DecArm::Dec32, DecArm::Serial, DecArm::Keys],
        m => anyhow::bail!("decode world bench mode {m} unknown (decw | decwf32 | decwserial | decwkeys | decwmix | decwvpg)"),
    };
    let world = a.world as usize;
    let rounds = crate::tp_xport::rounds_of(world)?;
    anyhow::ensure!(a.rank >= 0 && (a.rank as usize) < world, "--rank {} is outside world {world}", a.rank);
    anyhow::ensure!(a.peer_ips.len() >= world, "--peer-ips needs {world} rank-indexed IPs (got {})", a.peer_ips.len());
    let n = a.floats;
    anyhow::ensure!(n >= 4 && n % 4 == 0 && n * 4 <= 256 * 1024, "decode bench: floats must be a multiple of 4 and <= 64K");
    println!("=== --tp-reduce-bench rank {} world {} ({} exchange(s) per logical reduce) mode {} floats {} ({:.0} KiB) batches {} K2 blocks {} ===",
             a.rank, world, rounds, a.mode, n, n as f64 * 4.0 / 1024.0, a.reduces, a.blocks);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    let names = ["xq_ks_combine_f32", "xq_ks_combine_f32_k1l", "xq_tp_wait_add_dec", "xq_cvt_f32_f16", "xq_tp_wait_keys"];
    let ptx = Ptx::from_src(crate::exl3_bench::bench_ptx_src());
    dev.load_ptx(ptx, "tpgw", &names)?;
    let f = |nm: &str| dev.get_func("tpgw", nm).ok_or_else(|| anyhow::anyhow!("{nm} missing"));
    let (kc, kcl, k2d, kcv, kkeys) = (f(names[0])?, f(names[1])?, f(names[2])?, f(names[3])?, f(names[4])?);
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..world].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let mut link = TpLink::connect_nway(a.rank, world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream

    // R10, exactly as the engine's attach does it: pad the device epoch counter to a multiple of `rounds`
    // with tiny serial exchanges (a fresh ctx is at 0, so the pad is 0 in practice).
    let mut zero4 = dev.alloc_zeros::<f32>(4)?;
    dev.memset_zeros(&mut zero4)?;
    dev.synchronize()?;
    let e_boot = link.device_epoch();
    let pad = ((rounds as u64 - e_boot % rounds as u64) % rounds as u64) as usize;
    for _ in 0..pad {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    anyhow::ensure!(link.device_epoch() % rounds as u64 == 0, "round phase not aligned after padding (epoch {})", link.device_epoch());
    println!("[bench] link up (world {world}); device epoch {} -> {} ({pad} pad exchange(s)); first reduce runs round {} first",
             e_boot, link.device_epoch(), (link.device_epoch() + 1) % rounds as u64);

    // buffers (one slot per reduce of a batch: every reduce has its own operands)
    let mut s1 = dev.alloc_zeros::<f32>(NB * n)?;  let p1 = dev.alloc_zeros::<f32>(NB * n)?;  let o1 = dev.alloc_zeros::<u16>(NB * n)?;
    let mut s2 = dev.alloc_zeros::<f32>(NB * n)?;  let p2 = dev.alloc_zeros::<f32>(NB * n)?;  let o2 = dev.alloc_zeros::<f32>(NB * n)?;
    let mut s3 = dev.alloc_zeros::<f32>(NB * n)?;  let p3 = dev.alloc_zeros::<f32>(NB * n)?;  let o3 = dev.alloc_zeros::<u16>(NB * n)?;
    let mut kb = dev.alloc_zeros::<u64>(NB * MKEYS)?;
    let ki = dev.alloc_zeros::<i32>(NB * MKEYS)?;
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;
    let (ps1, pp1, po1) = (*s1.device_ptr() as u64, *p1.device_ptr() as u64, *o1.device_ptr() as u64);
    let (ps2, pp2, po2) = (*s2.device_ptr() as u64, *p2.device_ptr() as u64, *o2.device_ptr() as u64);
    let (ps3, pp3, po3) = (*s3.device_ptr() as u64, *p3.device_ptr() as u64, *o3.device_ptr() as u64);
    let (pkb, pki) = (*kb.device_ptr() as u64, *ki.device_ptr() as u64);
    // per-reduce shapes (the mix varies them; the single-arm modes use n / 16 rows throughout)
    let mix = a.mode == "decwmix";
    let ser_w = |j: usize| -> usize { if mix { (4 * (1 + (j * 131) % 1024)).min(n) } else { n } };
    let key_m = |j: usize| -> usize { if mix || a.mode == "decwkeys" { 1 + j % MKEYS } else { MKEYS } };
    let k2cfg = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let gcfg = |w: usize| LaunchConfig { grid_dim: ((w as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

    // One decode-path logical reduce of arm `arm`, reduce slot `j`: the engine's launch sequence.
    let launch_arm = |arm: DecArm, j: usize| -> anyhow::Result<()> {
        match arm {
            DecArm::Dec16 | DecArm::Dec32 => {
                let f16o = arm == DecArm::Dec16;
                let (s, p, o) = if f16o { (ps1, pp1, po1) } else { (ps2, pp2, po2) };
                let (s, p, o) = (s + (j * n * 4) as u64, p + (j * n * 4) as u64, o + (j * n * if f16o { 2 } else { 4 }) as u64);
                unsafe { kcl.clone().launch_on_stream(&stream, gcfg(n), (ctx, s, p, 1i32, n as i32, 1i32, arr))?; }
                let inter = crate::tp_xport::dec_intermediate(p, o, f16o);
                for x in crate::tp_xport::dec_exchanges(rounds) {
                    if !x.folded_k1 { crate::tp_xport::k1_serial(&kern, &stream, ctx, inter, n)?; }
                    let k = crate::tp_xport::dec_exchange_args(&x, p, false, o, f16o);
                    let io = (k.local_f16 as i32) | ((k.out_f16 as i32) << 1);
                    unsafe { k2d.clone().launch_on_stream(&stream, k2cfg, (ctx, k.out, k.local, n as i32, io, 0u64, 0u32, 0u64, 0u32))?; }
                }
            }
            DecArm::Serial => {
                let w = ser_w(j);
                let (s, p, o) = (ps3 + (j * n * 4) as u64, pp3 + (j * n * 4) as u64, po3 + (j * n * 2) as u64);
                unsafe { kc.clone().launch_on_stream(&stream, gcfg(w), (s, p, 1i32, w as i32, 1i32))?; }
                crate::tp_xport::reduce_serial(&kern, &stream, ctx, p, w, rounds)?;
                unsafe { kcv.clone().launch_on_stream(&stream, gcfg(w), (o, p, w as i64))?; }
            }
            DecArm::Keys => {
                let m = key_m(j);
                let mpad = m + (m & 1);
                let nbytes = 8 * mpad;
                let (kp, ip) = (pkb + (j * MKEYS * 8) as u64, pki + (j * MKEYS * 4) as u64);
                for _x in crate::tp_xport::dec_exchanges(rounds) {
                    crate::tp_xport::k1_serial(&kern, &stream, ctx, kp, nbytes / 4)?;
                    let c = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
                    unsafe { kkeys.clone().launch_on_stream(&stream, c, (ctx, kp, ip, m as i32, nbytes as i32, 0i32))?; }
                }
            }
        }
        Ok(())
    };

    let f16b = |v: f32| half::f16::from_f32(v).to_bits();
    let reps = a.reduces.max(1);
    let (mut times, mut bad, mut checked) = (Vec::new(), 0usize, 0usize);
    let mut host = vec![0f32; NB * n];
    let mut hkeys = vec![0u64; NB * MKEYS];
    for it in 0..reps {
        let salt = |arm: usize, j: usize| it * NB + j + arm * 1_000_000;
        // ---- fill (host, untimed) ----
        for (arm, buf, sl) in [(0usize, &mut s1, DecArm::Dec16), (1, &mut s2, DecArm::Dec32), (2, &mut s3, DecArm::Serial)] {
            if !active.contains(&sl) { continue; }
            for j in 0..NB { for i in 0..n { host[j * n + i] = fill_val(a.rank, i, salt(arm, j)); } }
            dev.htod_sync_copy_into(&host, buf)?;
        }
        if active.contains(&DecArm::Keys) {
            hkeys.iter_mut().for_each(|k| *k = 0);
            for j in 0..NB { for r in 0..key_m(j) { hkeys[j * MKEYS + r] = dec_key(a.rank, r, salt(3, j)); } }
            dev.htod_sync_copy_into(&hkeys, &mut kb)?;
        }
        dev.synchronize()?;
        let e_start = link.device_epoch();
        anyhow::ensure!(e_start % rounds as u64 == 0, "round phase lost before batch {it}: epoch {e_start}");
        // ---- timed batch: NB logical reduces per active arm, rotating arm order ----
        let mut e0 = vec![vec![0u64; NB]; 4];
        let mut seq = 0u64;
        let t0 = std::time::Instant::now();
        for j in 0..NB {
            let mut order = active.clone();
            let rot = j % order.len();
            order.rotate_left(rot);
            for &arm in &order {
                e0[arm as usize][j] = e_start + seq;
                launch_arm(arm, j)?;
                seq += rounds as u64;
            }
        }
        dev.synchronize()?;
        times.push(t0.elapsed().as_nanos() as u64 / NB as u64);
        let e_end = link.device_epoch();
        anyhow::ensure!(e_end == e_start + seq,
            "EPOCH ACCOUNTING: batch {it} consumed {} epochs, expected {seq} ({} logical reduces x {rounds} rounds)",
            e_end - e_start, seq / rounds as u64);
        if link.abort_status() != 0 { anyhow::bail!("transport ABORT status {} at batch {it}", link.abort_status()); }
        // ---- validate every reduce bitwise against the host reference ----
        let parts = |arm: usize, j: usize, w: usize| -> Vec<Vec<f32>> {
            (0..world).map(|r| (0..w).map(|i| fill_val(r as i32, i, salt(arm, j))).collect()).collect()
        };
        if active.contains(&DecArm::Dec16) {
            let got = dev.dtoh_sync_copy(&o1)?;
            for j in 0..NB {
                let want = crate::tp_xport::rd_tree_reference(world, e0[0][j], &parts(0, j, n))?;
                for i in 0..n {
                    checked += 1;
                    if got[j * n + i] != f16b(want[i]) {
                        if bad < 5 { println!("  MISMATCH decw batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got[j * n + i], f16b(want[i])); }
                        bad += 1;
                    }
                }
            }
        }
        if active.contains(&DecArm::Dec32) {
            let got = dev.dtoh_sync_copy(&o2)?;
            let local = dev.dtoh_sync_copy(&p2)?;
            for j in 0..NB {
                let want = crate::tp_xport::rd_tree_reference(world, e0[1][j], &parts(1, j, n))?;
                for i in 0..n {
                    checked += 1;
                    if got[j * n + i].to_bits() != want[i].to_bits() {
                        if bad < 5 { println!("  MISMATCH decwf32 batch {it} reduce {j} elem {i}: got {:e} want {:e}", got[j * n + i], want[i]); }
                        bad += 1;
                    }
                    // the screen invariant: `local` (the producer's partial) is not the intermediate and survives
                    checked += 1;
                    if local[j * n + i].to_bits() != fill_val(a.rank, i, salt(1, j)).to_bits() {
                        if bad < 5 { println!("  MISMATCH decwf32 batch {it} reduce {j} elem {i}: LOCAL partial was overwritten"); }
                        bad += 1;
                    }
                }
            }
        }
        if active.contains(&DecArm::Serial) {
            let got = dev.dtoh_sync_copy(&o3)?;
            for j in 0..NB {
                let w = ser_w(j);
                let want = crate::tp_xport::rd_tree_reference(world, e0[2][j], &parts(2, j, w))?;
                for i in 0..w {
                    checked += 1;
                    if got[j * n + i] != f16b(want[i]) {
                        if bad < 5 { println!("  MISMATCH decwserial batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got[j * n + i], f16b(want[i])); }
                        bad += 1;
                    }
                }
            }
        }
        if active.contains(&DecArm::Keys) {
            let ids = dev.dtoh_sync_copy(&ki)?;
            let merged = dev.dtoh_sync_copy(&kb)?;
            for j in 0..NB {
                for r in 0..key_m(j) {
                    let mx = (0..world as i32).map(|rk| dec_key(rk, r, salt(3, j))).max().unwrap();
                    let want_id = (0xFFFF_FFFFu32 - (mx & 0xFFFF_FFFF) as u32) as i32;
                    checked += 1;
                    if ids[j * MKEYS + r] != want_id || merged[j * MKEYS + r] != mx {
                        if bad < 5 { println!("  MISMATCH decwkeys batch {it} reduce {j} row {r}: id {} want {want_id}, key {:016x} want {mx:016x}", ids[j * MKEYS + r], merged[j * MKEYS + r]); }
                        bad += 1;
                    }
                }
            }
        }
    }
    let mut s = if times.len() > 1 { times[1..].to_vec() } else { times.clone() };   // batch 0 = cold start, validated only
    s.sort_unstable();
    let per = |p: f64| pct(&s, p) as f64 / 1e3;
    let unit = if active.len() > 1 { format!("mixed round of {} arms", active.len()) } else { "logical reduce".to_string() };
    println!("  {} per {unit}: p10 {:.2} p50 {:.2} p90 {:.2} us  ({} exchange(s) each => {:.2} us per exchange at p50)",
             a.mode, per(0.1), per(0.5), per(0.9), rounds * active.len(), per(0.5) / (rounds * active.len()) as f64);
    println!("  tail fires {}  abort {}  device epoch {} gpu_ready {}", link.tail_fires(), link.abort_status(), link.device_epoch(), link.gpu_ready());
    println!("  EPOCH ACCOUNTING: every batch consumed exactly {rounds} epoch(s) per logical reduce => OK");
    println!("  VALIDATION: {bad} mismatching elements of {checked} checked over {reps} x {NB} x {} arm(s) => {}",
             active.len(), if bad == 0 && checked > 0 { "BITWISE_OK" } else { "FAIL" });
    // quiesce: one tiny serial reduce after the verdict so no rank leaves while a peer still waits on its last send
    for _ in 0..rounds {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(300));
    if bad != 0 || checked == 0 { anyhow::bail!("decode world bench validation FAILED"); }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// TP-4F2: `--tp-reduce-bench --mode decws --world 4` — the SINGLE-STAGE decode reduce (`--tp-reduce single`) against
// the two-stage rd reduce, on the REAL transports, no model: ONE process per rank brings up the rd link (`--port`,
// proxy core `--proxy-core`) AND the dedicated uniform one-shot ctx (`--port` + 40, proxy core `--proxy-core` - 2,
// slot capacity 256 KiB as the engine builds it). Every iteration reduces the SAME per-rank partials through both
// arms (the engine's launch sequences: the producer-folded K1 + rounds x xq_tp_wait_add_dec, vs the folded K1 on the
// single ctx + ONE xq_tp_wait_add_dec_single), batches of 96 logical reduces timed per arm in an ABBA alternation
// (the first batch is validated, not timed). Checked on EVERY rank, bitwise: rd output == f16(rd_tree_reference),
// single output == f16(single_tree_reference), and single == rd element for element. The epoch counters must advance
// by exactly `rounds` per rd reduce and exactly 1 per single reduce (the single ctx has its own phase-free counter).
// ---------------------------------------------------------------------------------------------
fn run_reduce_single_nway(a: DecWorldArgs) -> anyhow::Result<()> {
    const NB: usize = 96;
    let world = a.world as usize;
    anyhow::ensure!(world == 4, "--mode decws is the world-4 single-stage bench (got --world {world})");
    let rounds = crate::tp_xport::rounds_of(world)?;
    anyhow::ensure!(a.rank >= 0 && (a.rank as usize) < world, "--rank {} is outside world {world}", a.rank);
    anyhow::ensure!(a.peer_ips.len() >= world, "--peer-ips needs {world} rank-indexed IPs (got {})", a.peer_ips.len());
    let n = a.floats;
    anyhow::ensure!(n >= 4 && n % 4 == 0 && n * 4 <= 256 * 1024, "decws bench: floats must be a multiple of 4 and <= 64K");
    println!("=== --tp-reduce-bench rank {} world {world} mode decws (rd: {rounds} exchanges; single: 1) floats {n} ({:.0} KiB) batches {} K2 blocks {} ===",
             a.rank, n as f64 * 4.0 / 1024.0, a.reduces, a.blocks);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    let names = ["xq_ks_combine_f32_k1l", "xq_tp_wait_add_dec", "xq_tp_wait_add_dec_single"];
    let ptx = Ptx::from_src(crate::exl3_bench::bench_ptx_src());
    dev.load_ptx(ptx, "tpsw", &names)?;
    let f = |nm: &str| dev.get_func("tpsw", nm).ok_or_else(|| anyhow::anyhow!("{nm} missing"));
    let (kcl, k2d, k2s) = (f(names[0])?, f(names[1])?, f(names[2])?);
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..world].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    // rail-1-style rd link first (every rank does the same sequence, so the handshakes pair), then the uniform ctx
    let mut link = TpLink::connect_nway(a.rank, world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let sport = a.port + 40;
    let mut slink = TpLink::connect_nway_uniform(a.rank, world as i32, &ips, sport, &a.dev, a.gid, 256 * 1024)?;
    slink.set_payload(2560 * 4, true)?;
    let sctx = slink.ctx_device_ptr();
    let slink = std::mem::ManuallyDrop::new(slink);
    net::spawn_proxy_aux(slink.ctx_addr(), a.proxy_core - 2);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream

    // R10 on the rd link exactly as the engine's attach does it (a fresh ctx is at epoch 0: pad 0)
    let mut zero4 = dev.alloc_zeros::<f32>(4)?;
    dev.memset_zeros(&mut zero4)?;
    dev.synchronize()?;
    let e_boot = link.device_epoch();
    let pad = ((rounds as u64 - e_boot % rounds as u64) % rounds as u64) as usize;
    for _ in 0..pad {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    anyhow::ensure!(link.device_epoch() % rounds as u64 == 0, "round phase not aligned after padding (epoch {})", link.device_epoch());
    println!("[bench] rd link + single-stage ctx up (world {world}); rd device epoch {} (pad {pad}), single ctx epoch {}",
             link.device_epoch(), slink.device_epoch());

    let mut ws = dev.alloc_zeros::<f32>(NB * n)?;
    let pa = dev.alloc_zeros::<f32>(NB * n)?;  let oa = dev.alloc_zeros::<u16>(NB * n)?;   // rd arm: partial, f16 out
    let ps = dev.alloc_zeros::<f32>(NB * n)?;  let os = dev.alloc_zeros::<u16>(NB * n)?;   // single arm
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;
    let (pws, ppa, poa, pps, pos) = (*ws.device_ptr() as u64, *pa.device_ptr() as u64, *oa.device_ptr() as u64,
                                     *ps.device_ptr() as u64, *os.device_ptr() as u64);
    let k2cfg = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let gcfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

    let launch_rd = |j: usize| -> anyhow::Result<()> {
        let (s, p, o) = (pws + (j * n * 4) as u64, ppa + (j * n * 4) as u64, poa + (j * n * 2) as u64);
        unsafe { kcl.clone().launch_on_stream(&stream, gcfg, (ctx, s, p, 1i32, n as i32, 1i32, arr))?; }
        let inter = crate::tp_xport::dec_intermediate(p, o, true);
        for x in crate::tp_xport::dec_exchanges(rounds) {
            if !x.folded_k1 { crate::tp_xport::k1_serial(&kern, &stream, ctx, inter, n)?; }
            let k = crate::tp_xport::dec_exchange_args(&x, p, false, o, true);
            let io = (k.local_f16 as i32) | ((k.out_f16 as i32) << 1);
            unsafe { k2d.clone().launch_on_stream(&stream, k2cfg, (ctx, k.out, k.local, n as i32, io, 0u64, 0u32, 0u64, 0u32))?; }
        }
        Ok(())
    };
    let launch_single = |j: usize| -> anyhow::Result<()> {
        let (s, p, o) = (pws + (j * n * 4) as u64, pps + (j * n * 4) as u64, pos + (j * n * 2) as u64);
        unsafe { kcl.clone().launch_on_stream(&stream, gcfg, (sctx, s, p, 1i32, n as i32, 1i32, arr))?; }
        unsafe { k2s.clone().launch_on_stream(&stream, k2cfg, (sctx, o, p, n as i32, 0u64, 0u32, 0u64, 0u32))?; }
        Ok(())
    };

    let f16b = |v: f32| half::f16::from_f32(v).to_bits();
    let reps = a.reduces.max(2);
    let (mut t_rd, mut t_si) = (Vec::new(), Vec::new());
    let (mut bad, mut checked, mut cross) = (0usize, 0usize, 0usize);
    let mut host = vec![0f32; NB * n];
    for it in 0..reps {
        let salt = |j: usize| it * NB + j;
        for j in 0..NB { for i in 0..n { host[j * n + i] = fill_val(a.rank, i, salt(j)); } }
        dev.htod_sync_copy_into(&host, &mut ws)?;
        dev.synchronize()?;
        let (e_rd0, e_si0) = (link.device_epoch(), slink.device_epoch());
        anyhow::ensure!(e_rd0 % rounds as u64 == 0, "rd round phase lost before batch {it}: epoch {e_rd0}");
        let rd_first = it % 2 == 0;      // ABBA: alternate which arm runs first in a batch pair
        let run_arm = |single: bool| -> anyhow::Result<u64> {
            let t0 = std::time::Instant::now();
            for j in 0..NB { if single { launch_single(j)?; } else { launch_rd(j)?; } }
            dev.synchronize()?;
            Ok(t0.elapsed().as_nanos() as u64 / NB as u64)
        };
        if rd_first { t_rd.push(run_arm(false)?); t_si.push(run_arm(true)?); }
        else { t_si.push(run_arm(true)?); t_rd.push(run_arm(false)?); }
        let (e_rd1, e_si1) = (link.device_epoch(), slink.device_epoch());
        anyhow::ensure!(e_rd1 == e_rd0 + (NB * rounds) as u64, "EPOCH ACCOUNTING (rd): batch {it} consumed {} epochs, expected {}", e_rd1 - e_rd0, NB * rounds);
        anyhow::ensure!(e_si1 == e_si0 + NB as u64, "EPOCH ACCOUNTING (single): batch {it} consumed {} epochs, expected {NB}", e_si1 - e_si0);
        if link.abort_status() != 0 || slink.abort_status() != 0 {
            anyhow::bail!("transport ABORT (rd {}, single {}) at batch {it}", link.abort_status(), slink.abort_status());
        }
        // ---- validate every reduce bitwise (on this rank) ----
        let got_rd = dev.dtoh_sync_copy(&oa)?;
        let got_si = dev.dtoh_sync_copy(&os)?;
        for j in 0..NB {
            let parts: Vec<Vec<f32>> = (0..world).map(|r| (0..n).map(|i| fill_val(r as i32, i, salt(j))).collect()).collect();
            // the rd arm of reduce j ran as the (j)th rd reduce of the batch: start epoch e_rd0 + j * rounds (even)
            let want_rd = crate::tp_xport::rd_tree_reference(world, e_rd0 + (j * rounds) as u64, &parts)?;
            let want_si = crate::tp_xport::single_tree_reference(&parts)?;
            for i in 0..n {
                checked += 2;
                if got_rd[j * n + i] != f16b(want_rd[i]) {
                    if bad < 5 { println!("  MISMATCH rd batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got_rd[j * n + i], f16b(want_rd[i])); }
                    bad += 1;
                }
                if got_si[j * n + i] != f16b(want_si[i]) {
                    if bad < 5 { println!("  MISMATCH single batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got_si[j * n + i], f16b(want_si[i])); }
                    bad += 1;
                }
                if got_si[j * n + i] != got_rd[j * n + i] { cross += 1; }
            }
        }
    }
    let stat = |t: &Vec<u64>| { let mut s = if t.len() > 1 { t[1..].to_vec() } else { t.clone() }; s.sort_unstable(); s };
    let (sr, ss) = (stat(&t_rd), stat(&t_si));
    let per = |s: &Vec<u64>, p: f64| pct(s, p) as f64 / 1e3;
    println!("  rd     per logical reduce ({rounds} exchanges): p10 {:.2} p50 {:.2} p90 {:.2} us", per(&sr, 0.1), per(&sr, 0.5), per(&sr, 0.9));
    println!("  single per logical reduce (1 exchange)  : p10 {:.2} p50 {:.2} p90 {:.2} us   => delta p50 {:+.2} us per reduce ({:+.2} ms per 96-reduce round)",
             per(&ss, 0.1), per(&ss, 0.5), per(&ss, 0.9), per(&ss, 0.5) - per(&sr, 0.5), (per(&ss, 0.5) - per(&sr, 0.5)) * 96.0 / 1e3);
    println!("  rd link: tail fires {} abort {} epoch {}   single ctx: tail fires {} abort {} epoch {} gpu_ready {}",
             link.tail_fires(), link.abort_status(), link.device_epoch(), slink.tail_fires(), slink.abort_status(), slink.device_epoch(), slink.gpu_ready());
    println!("  EPOCH ACCOUNTING: rd consumed exactly {rounds} epochs and single exactly 1 per logical reduce => OK");
    println!("  VALIDATION: {bad} mismatching elements of {checked} checked (rd vs rd_tree_reference, single vs single_tree_reference) over {reps} x {NB} reduces; \
              single-vs-rd cross mismatches {cross} => {}", if bad == 0 && cross == 0 && checked > 0 { "BITWISE_OK" } else { "FAIL" });
    // quiesce so no rank leaves while a peer still waits on its last send
    for _ in 0..rounds {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(300));
    if bad != 0 || cross != 0 || checked == 0 { anyhow::bail!("single-stage decode bench validation FAILED"); }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// TP-4H2: `--tp-reduce-bench --mode decwvpg --world 4` — the WORLD-4 vocab-parallel row all-gather
// (`xq_vp_gather4_k1/k2`, the sampled / penalized / ratio-rule tail of `--tp-vp-sampled`) on the REAL
// transport, no model. It launches exactly the engine's sequence (`tp_vpgather::gather4_plan`, the same
// kernels, the same arguments as `vp_gather_rows_w4`) over synthetic, POSITION-DEPENDENT shard rows
// (`tp_vpgather::ref_half`, salted per gather so a stale ring slot can never match) and checks, on EVERY
// rank, that every assembled logits row is bit for bit the four shards in place. No arithmetic is involved
// (ks = 1: the gather is a bit copy), so the reference is exact equality. The device epoch counter is read
// back after every timed batch and must have advanced by EXACTLY `epochs_per_gather(m)` per gather (plus
// `rounds` per interleaved serial exchange in batch 0) — the hardware proof of the R10 phase discipline for
// this arm. Batch 0 of every width is validated but excluded from the percentiles; it also interleaves tiny
// serial exchanges between gathers (the decode arms' phase) so a gather that consumed a wrong number of
// epochs corrupts a LATER gather's partner schedule and fails.
// Widths: 1 2 3 4 5 8 9 12 13 16 (every group count and both partial-group shapes). Per gather: NG = 8
// gathers per timed batch, each with its own shard and logits operands.
// ---------------------------------------------------------------------------------------------
pub fn run_vp_gather_nway(a: DecWorldArgs) -> anyhow::Result<()> {
    use crate::tp_vpgather as g;
    const NSH: usize = 62_080;
    const V: usize = NSH * g::WORLD;
    const NG: usize = 8;
    const WIDTHS: [usize; 10] = [1, 2, 3, 4, 5, 8, 9, 12, 13, 16];
    let world = a.world as usize;
    anyhow::ensure!(world == g::WORLD, "--mode decwvpg is the world-4 row all-gather bench (got --world {world})");
    let rounds = crate::tp_xport::rounds_of(world)?;
    anyhow::ensure!(rounds == g::ROUNDS, "rounds {rounds} != {}", g::ROUNDS);
    anyhow::ensure!(a.rank >= 0 && (a.rank as usize) < world, "--rank {} is outside world {world}", a.rank);
    anyhow::ensure!(a.peer_ips.len() >= world, "--peer-ips needs {world} rank-indexed IPs (got {})", a.peer_ips.len());
    anyhow::ensure!(g::slot_fits(g::GROUP_ROWS, NSH, crate::tp::TP_SLOT_BYTES), "a 4-row group does not fit one ring slot");
    let rank = a.rank as usize;
    println!("=== --tp-reduce-bench rank {} world {} ({} exchange(s) per 4-row group) mode {} n_sh {} V {} gathers/batch {} batches {} ===",
             a.rank, world, rounds, a.mode, NSH, V, NG, a.reduces);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    let names = ["xq_vp_gather4_k1", "xq_vp_gather4_k2"];
    let ptx = Ptx::from_src(crate::exl3_bench::bench_ptx_src());
    dev.load_ptx(ptx, "tpvg", &names)?;
    let f = |nm: &str| dev.get_func("tpvg", nm).ok_or_else(|| anyhow::anyhow!("{nm} missing"));
    let (k1, k2) = (f(names[0])?, f(names[1])?);
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..world].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let mut link = TpLink::connect_nway(a.rank, world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream

    // R10, exactly as the engine's attach does it (a fresh ctx is at 0, so the pad is 0 in practice)
    let mut zero4 = dev.alloc_zeros::<f32>(4)?;
    dev.memset_zeros(&mut zero4)?;
    dev.synchronize()?;
    let zp = *zero4.device_ptr() as u64;
    let e_boot = link.device_epoch();
    let pad = ((rounds as u64 - e_boot % rounds as u64) % rounds as u64) as usize;
    for _ in 0..pad {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, zp, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, zp, 4)?;
    }
    dev.synchronize()?;
    anyhow::ensure!(link.device_epoch() % rounds as u64 == 0, "round phase not aligned after padding (epoch {})", link.device_epoch());
    println!("[bench] link up (world {world}); device epoch {} -> {} ({pad} pad exchange(s)); first gather runs round {} first",
             e_boot, link.device_epoch(), (link.device_epoch() + 1) % rounds as u64);
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;

    let reps = a.reduces.max(2);
    let (mut bad, mut checked) = (0usize, 0usize);
    let mut summary: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
    for (mi, &m) in WIDTHS.iter().enumerate() {
        let eps = g::epochs_per_gather(m);
        let plan = g::gather4_plan(m, NSH);
        let mut shard = dev.alloc_zeros::<u16>(NG * m * NSH)?;
        let mut logits = dev.alloc_zeros::<u16>(NG * m * V)?;
        let (psh, plg) = (*shard.device_ptr() as u64, *logits.device_ptr() as u64);
        let poison = vec![0x7E7Eu16; NG * m * V];
        let mut hsh = vec![0u16; NG * m * NSH];
        let mut times: Vec<u64> = Vec::new();
        // the synthetic row id of (batch it, gather j, row r): unique across every width and batch
        let rowid = |it: usize, j: usize, r: usize| ((mi * reps + it) * NG + j) * g::MAX_ROWS + r;
        for it in 0..reps {
            // ---- fill (host, untimed): this rank's shard rows; poison every logits column ----
            for j in 0..NG {
                for r in 0..m {
                    let row = rowid(it, j, r);
                    let o = (j * m + r) * NSH;
                    for c in 0..NSH { hsh[o + c] = g::ref_half(rank, row, rank * NSH + c); }
                }
            }
            dev.htod_sync_copy_into(&hsh, &mut shard)?;
            dev.htod_sync_copy_into(&poison, &mut logits)?;
            dev.synchronize()?;
            let e_start = link.device_epoch();
            anyhow::ensure!(e_start % rounds as u64 == 0, "round phase lost before width {m} batch {it}: epoch {e_start}");
            // ---- timed batch: NG gathers back to back (batch 0 interleaves serial exchanges: untimed) ----
            let inter = it == 0;
            let mut extra = 0u64;
            let t0 = std::time::Instant::now();
            for j in 0..NG {
                let (sb, lb) = (psh + (j * m * NSH * 2) as u64, plg + (j * m * V * 2) as u64);
                for la in &plan {
                    let cfg = LaunchConfig { grid_dim: (la.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    match la.half {
                        g::Half::K1 => unsafe {
                            k1.clone().launch_on_stream(&stream, cfg,
                                (ctx, sb, lb, V as i64, NSH as i32, la.r0 as i32, la.nr as i32, la.stage as i32, arr))?;
                        },
                        g::Half::K2 => unsafe {
                            k2.clone().launch_on_stream(&stream, cfg,
                                (ctx, lb, V as i64, NSH as i32, la.r0 as i32, la.nr as i32, la.stage as i32, la.lag, 0i32))?;
                        },
                    }
                }
                if inter && j % 2 == 1 {
                    for _ in 0..rounds {
                        crate::tp_xport::k1_serial(&kern, &stream, ctx, zp, 4)?;
                        crate::tp_xport::k2_serial(&kern, &stream, ctx, zp, 4)?;
                    }
                    extra += rounds as u64;
                }
            }
            dev.synchronize()?;
            if !inter { times.push(t0.elapsed().as_nanos() as u64 / NG as u64); }
            let want = (NG * eps) as u64 + extra;
            let e_end = link.device_epoch();
            anyhow::ensure!(e_end == e_start + want,
                "EPOCH ACCOUNTING: width {m} batch {it} consumed {} epochs, expected {want} ({NG} gathers x {eps} + {extra} interleaved)",
                e_end - e_start);
            if link.abort_status() != 0 { anyhow::bail!("transport ABORT status {} at width {m} batch {it}", link.abort_status()); }
            // ---- validate: every assembled row is the four shards in place, bit for bit ----
            let got = dev.dtoh_sync_copy(&logits)?;
            let mut row_bad = 0usize;
            for j in 0..NG {
                for r in 0..m {
                    let row = rowid(it, j, r);
                    let o = (j * m + r) * V;
                    for s in 0..world {
                        for c in 0..NSH {
                            checked += 1;
                            let w = g::ref_half(s, row, s * NSH + c);
                            if got[o + s * NSH + c] != w {
                                if bad + row_bad < 5 {
                                    println!("  MISMATCH decwvpg width {m} batch {it} gather {j} row {r} shard {s} col {c}: got {:04x} want {w:04x}",
                                             got[o + s * NSH + c]);
                                }
                                row_bad += 1;
                            }
                        }
                    }
                }
            }
            bad += row_bad;
        }
        let mut s = times.clone();   // batch 0 (cold, interleaved) is validated only
        s.sort_unstable();
        let per = |p: f64| pct(&s, p) as f64 / 1e3;
        println!("  decwvpg m={m:<2}: {} group(s), {eps} epoch(s) per gather, payload {} B (+8 tail) per last-stage epoch: \
                  p10 {:.1} p50 {:.1} p90 {:.1} us per gather ({:.1} us per epoch at p50)",
                 m.div_ceil(g::GROUP_ROWS), g::payload_bytes(g::GROUP_ROWS.min(m), NSH, 1), per(0.1), per(0.5), per(0.9), per(0.5) / eps as f64);
        summary.push((m, eps, per(0.1), per(0.5), per(0.9)));
    }
    println!("  VPG_US (p50 us per gather): {}",
             summary.iter().map(|&(m, _, _, p50, _)| format!("m={m}:{p50:.1}")).collect::<Vec<_>>().join(" "));
    println!("  tail fires {}  abort {}  device epoch {} gpu_ready {}", link.tail_fires(), link.abort_status(), link.device_epoch(), link.gpu_ready());
    println!("  EPOCH ACCOUNTING: every batch consumed exactly epochs_per_gather(m) x gathers (+ rounds per interleaved serial exchange) => OK");
    println!("  VALIDATION: {bad} mismatching elements of {checked} checked over {reps} batches x {NG} gathers x widths {:?} => {}",
             WIDTHS, if bad == 0 && checked > 0 { "BITWISE_OK" } else { "FAIL" });
    // quiesce: one tiny serial reduce after the verdict so no rank leaves while a peer still waits on its last send
    for _ in 0..rounds {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, zp, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, zp, 4)?;
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(300));
    if bad != 0 || checked == 0 { anyhow::bail!("world-4 row all-gather bench validation FAILED"); }
    Ok(())
}
