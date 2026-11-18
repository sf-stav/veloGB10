//! Grouped A1, LM-head, split-K, WP21 and dense-fusion probes.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// MTPAGREE1 (2026-10-28): the reference launches below drive k2-keyed kernels (exl3_hmma_gemm, _ks, _ks_x2, _ks_fx —
// their switch takes 2*K since W3, 2026-10-18); they kept passing `bits`, so on a 4-bit module the reference ran the
// K=2.0 body and on a 6-bit module the K=3 body, and --probe-exl3-binv reported false kernel-equivalence FAILs. The
// grouped / wide kernels stay bits-keyed and keep `bits`.
// ---------------------------------------------------------------------------
// S-A3-n F3: the grouped expert kernel's A-once body must be BITWISE equal to the
// barrier body (exl3_hmma_gemm) on the same trellis + activations, every row,
// every width that takes it. One expert (offs = [0], esel = [1]).
// ---------------------------------------------------------------------------
pub(super) fn probe_grouped_a1(dev: &std::sync::Arc<CudaDevice>, d_tr: &CudaSlice<u16>, k: usize, n: usize,
                    k2: usize) -> Result<()> {
    if k % 8 != 0 || n % 128 != 0 { return Ok(()); }
    // K35: the grouped kernel takes the width CODE (integer K, or EXL3_FRAC_BITS(KA) for K = KA + 0.5); the
    // m16 reference is k2-keyed. Widths whose A-once tile exceeds 48 KiB run the barrier body inside the
    // grouped kernel (dyn smem 0) — both bodies must equal the reference, so no width is skipped.
    let bits = crate::exl3_forward::exl3_kernel_bits(k2);
    let frac = k2 % 2 == 1;
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    let d_offs = dev.htod_sync_copy(&[0u64])?;
    let d_esel = dev.htod_sync_copy(&[1i32])?;
    let mut done = Vec::new();
    let mut barrier_widths = Vec::new();
    let mut entries = Vec::new();
    for m in [1usize, 2, 3, 4, 6, 8, 9, 10, 12, 16] {
        let smem = m * (k + 8) * 2;
        let smem = if smem > 48 * 1024 { barrier_widths.push(m); 0 } else { smem };
        // ISSUES 231 (2026-11-13): the probe must launch the entry the SERVER launches
        // (crate::exl3_forward::grouped_fn), not a hardcoded name. K = 8 has its own entry
        // (xq_gemm_grouped_b8, K8EXP) because xq_gemm_grouped_xh has no case 8 — probing xh at
        // k2 = 16 compared the reference against a kernel that wrote NOTHING (the switch's
        // `default: break`), which reads as "2560/2560 mismatches" but is a stale harness, not a
        // numeric drift. Dispatching through grouped_fn makes this leg exercise the served path.
        let name = crate::exl3_forward::grouped_fn(m, k, bits);
        let f_grp = dev.get_func(MODULE, name).with_context(|| format!("grouped entry {name}"))?;
        if !entries.contains(&name) { entries.push(name); }
        let xh = synth_x(m, k, 0xBEEF ^ m as u32);
        let d_xh = dev.htod_sync_copy(&xh)?;
        let mut y_old = dev.htod_sync_copy(&vec![0u16; m * n])?;
        let mut y_grp = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?; // poisoned: an unwritten tile is a mismatch
        let cfg = |sm: u32| LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: sm };
        unsafe {
            f_old.clone().launch(cfg(0), (d_tr, &d_xh, &mut y_old, m as i32, k as i32, n as i32, k2 as i32))?;
            f_grp.clone().launch(cfg(smem as u32), (d_tr, &d_offs, &d_xh, &mut y_grp, m as i32, k as i32, n as i32, bits as i32, &d_esel))?;
        }
        dev.synchronize()?;
        let a = dev.dtoh_sync_copy(&y_old)?;
        let b = dev.dtoh_sync_copy(&y_grp)?;
        if a.iter().filter(|v| **v != 0).count() < m * n / 2 { bail!("EXL3-GROUPED-A1 FAIL: implausible reference (m={m})"); }
        let mism = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        if mism > 0 { bail!("EXL3-GROUPED-A1 FAIL: m={m} K={k} N={n} k2={k2} (bits code {bits}): {mism} mismatches vs the k2-keyed m16 reference"); }
        if bits == 3 && !frac && smem > 0 {   // the dedicated A-once entries need the A-once tile (no barrier twin)
            // S-A3-o O3: the dedicated 3-bit entry must equal the barrier body too.
            let f_b3 = dev.get_func(MODULE, "xq_gemm_grouped_a1b3").context("xq_gemm_grouped_a1b3")?;
            let mut y_b3 = dev.htod_sync_copy(&vec![0u16; m * n])?;
            unsafe { f_b3.clone().launch(cfg(smem as u32), (d_tr, &d_offs, &d_xh, &mut y_b3, m as i32, k as i32, n as i32, bits as i32, &d_esel))?; }
            dev.synchronize()?;
            let c = dev.dtoh_sync_copy(&y_b3)?;
            let mism = a.iter().zip(c.iter()).filter(|(x, y)| x != y).count();
            if mism > 0 { bail!("EXL3-GROUPED-A1B3 FAIL: m={m} K={k} N={n}: {mism} mismatches vs the barrier body"); }
            // A5 WP20 rung 1: the word-diet twin (2 ring words per lane per k16 step, no SHFL).
            let f_wd = dev.get_func(MODULE, "xq_gemm_grouped_a1b3_wd").context("xq_gemm_grouped_a1b3_wd")?;
            let mut y_wd = dev.htod_sync_copy(&vec![0u16; m * n])?;
            unsafe { f_wd.clone().launch(cfg(smem as u32), (d_tr, &d_offs, &d_xh, &mut y_wd, m as i32, k as i32, n as i32, bits as i32, &d_esel))?; }
            dev.synchronize()?;
            let c = dev.dtoh_sync_copy(&y_wd)?;
            let mism = a.iter().zip(c.iter()).filter(|(x, y)| x != y).count();
            if mism > 0 { bail!("EXL3-GROUPED-A1B3-WD FAIL: m={m} K={k} N={n}: {mism} mismatches vs the barrier body"); }
        }
        done.push(m);
    }
    println!("    grouped {}body{} (bits code {bits}): PASS (bitwise == k2-keyed m16 reference, widths {done:?}; barrier body at {barrier_widths:?}; served entry(s) {entries:?})",
             if frac { "fractional (HALF) a1 + barrier " } else { "A-once " },
             if bits == 3 { " + a1b3 entry + a1b3_wd (WP20 diet)" } else { "" });
    Ok(())
}

// ---------------------------------------------------------------------------
// W3/LMH gate: the persistent lm_head GEMM (exl3_lmh_*, launched through the SERVED launcher
// crate::exl3_forward::lmh_gemm_launch) must be BITWISE equal to exl3_hmma_gemm (y_raw) and to
// exl3_hmma_gemm + exl3_had_svh (fused epilogue -> logits) at every m in `ms`, for every ring
// depth in `nsts`. Output buffers are poisoned (0xFFFF) so an unwritten element is a mismatch.
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub(super) fn probe_lmh(dev: &std::sync::Arc<CudaDevice>, d_tr: &CudaSlice<u16>, d_svh: &CudaSlice<u16>,
             k: usize, n: usize, bits: usize, ms: &[usize], nsts: &[usize], tag: &str) -> Result<()> {
    // The launcher's static eligibility (lmh_gemm_launch): shapes outside it keep the old kernel
    // by design — skip them here, so a refusal below is always a real (resource/PTX) failure.
    if k % (16 * crate::exl3_forward::LMH_KSTEP) != 0 || k % 128 != 0 || n % 128 != 0 || !(3..=6).contains(&bits) {
        println!("    W3/LMH [{tag}]: skipped (K={k} N={n} b={bits} outside the persistent kernel's shape class)");
        return Ok(());
    }
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    let f_svh = dev.get_func(MODULE, "exl3_had_svh").context("exl3_had_svh")?;
    let stream = fork_blocking_stream(dev);
    let mut checked = 0usize;
    for &m in ms {
        let xh = synth_x(m, k, 0x1A4D ^ (m as u32).wrapping_mul(2654435761));
        let d_xh = dev.htod_sync_copy(&xh)?;
        let mut yr_old = dev.htod_sync_copy(&vec![0u16; m * n])?;
        let mut y_old = dev.htod_sync_copy(&vec![0u16; m * n])?;
        unsafe {
            f_old.clone().launch_on_stream(&stream,
                LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (d_tr, &d_xh, &mut yr_old, m as i32, k as i32, n as i32, (2 * bits) as i32))?;
            f_svh.clone().launch_on_stream(&stream,
                LaunchConfig { grid_dim: ((m * n / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                (&yr_old, d_svh, &mut y_old, n as i32))?;
        }
        dev.synchronize()?;
        let r_old = dev.dtoh_sync_copy(&yr_old)?;
        let v_old = dev.dtoh_sync_copy(&y_old)?;
        if r_old.iter().filter(|v| **v != 0).count() < m * n / 2 {
            bail!("W3/LMH FAIL [{tag}]: implausible reference (m={m}: y_raw mostly zero)");
        }
        for &nst in nsts {
            for fuse in [false, true] {
                let out = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
                let ok = crate::exl3_forward::lmh_gemm_launch(dev, stream.stream, *d_tr.device_ptr(),
                    *d_xh.device_ptr(), *out.device_ptr(), *d_svh.device_ptr(), m, k, n, bits, nst, fuse)?;
                if !ok {
                    bail!("W3/LMH FAIL [{tag}]: launcher refused m={m} K={k} N={n} b={bits} nst={nst}");
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&out)?;
                let refv = if fuse { &v_old } else { &r_old };
                let mut mism = 0usize;
                let mut first = None;
                for i in 0..m * n {
                    if got[i] != refv[i] {
                        mism += 1;
                        if first.is_none() { first = Some((i / n, i % n, got[i], refv[i])); }
                    }
                }
                if mism > 0 {
                    bail!("W3/LMH FAIL [{tag}]: m={m} K={k} N={n} b={bits} nst={nst} fused={fuse}: \
                           {mism} of {} mismatches, first (row,col,new,old) {first:?}", m * n);
                }
                checked += 1;
            }
        }
    }
    println!("    W3/LMH [{tag}]: PASS (bitwise == exl3_hmma_gemm / +exl3_had_svh; K={k} N={n} b={bits}, \
              m {ms:?}, nst {nsts:?}, both epilogues, {checked} launches)");
    Ok(())
}

// ---------------------------------------------------------------------------
// w3-PSK (T4e): the persistent split-K chain GEMM must be BITWISE equal to the one-CTA-per-item
// grid it replaces — every fp32 partial ws[slab][row][col] — for EVERY chain shape this pack
// splits (non-expert, non-lm_head modules with chain_ks > 1), every width M = 1..16, a spread of
// grid sizes G (1, odd, 48-multiples, the built-in default, items-1, items, > items) and trellis
// prefetch depths 0/1/2: exl3_hmma_gemm_ks_p vs exl3_hmma_gemm_ks, and exl3_hmma_gemm_ks_p_x2 vs
// exl3_hmma_gemm_ks_x2 (two distinct same-shape modules when the pack has them). PSK workspaces
// are NaN-poisoned before every launch (an unwritten partial cannot pass), the reference must
// be plausible (finite, mostly nonzero), and y_raw through the UNCHANGED exl3_ks_combine keeps
// row 0 bitwise across widths (batch invariance).
// ---------------------------------------------------------------------------
pub(super) fn probe_ks_p(dev: &std::sync::Arc<CudaDevice>, pack: &Exl3Pack) -> Result<()> {
    use cudarc::driver::sys;
    use std::collections::BTreeMap;
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm_ks").context("exl3_hmma_gemm_ks")?;
    let f_old2 = dev.get_func(MODULE, "exl3_hmma_gemm_ks_x2").context("exl3_hmma_gemm_ks_x2")?;
    let f_new = dev.get_func(MODULE, "exl3_hmma_gemm_ks_p").context("exl3_hmma_gemm_ks_p (stale PTX?)")?;
    let f_new2 = dev.get_func(MODULE, "exl3_hmma_gemm_ks_p_x2").context("exl3_hmma_gemm_ks_p_x2 (stale PTX?)")?;
    let f_comb = dev.get_func(MODULE, "exl3_ks_combine").context("exl3_ks_combine")?;
    let sms = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?.max(1) as u32;
    let slots1 = f_new.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? * sms;
    let slots2 = f_new2.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? * sms;
    if slots1 == 0 || slots2 == 0 {
        bail!("EXL3-KS-P FAIL: zero occupancy for the PSK kernels (single {slots1}, pair {slots2})");
    }
    // distinct (K, N, bits) chain shapes -> up to two modules (pair members)
    let mut shapes: BTreeMap<(usize, usize, usize), Vec<&ModuleMeta>> = BTreeMap::new();
    let mut frac_skips = 0usize;
    let mut wide_skips = 0usize;
    for md in &pack.modules {
        if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") {
            continue;
        }
        if md.k % 128 != 0 || md.n % 128 != 0 { continue; }
        if crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) <= 1 { continue; }
        // P7 B8: PSK legs are INTEGER-width only (bodies bits 2..=6; the host falls back
        // to the chain for anything else — psk_grid guard, B6). Fractional k2 skips here
        // with a line instead of a false FAIL; the key carries TRUE k2 so grouping never
        // mixes 1.5 with 2.0.
        if md.k2 % 2 == 1 {
            frac_skips += 1;
            continue;
        }
        // PCKG-ALL (2026-10-30): persistent bodies are INTEGER 2..=6 (k2 4..=12). k2 > 12
        // (e.g. 16 = the 8-bit MTP-head quad) has no persistent body and psk_grid refuses
        // it at serve time; skip it here with a visible line instead of a false FAIL
        // (the probe used to force the kernel and read NaN no-op partials at b16).
        if md.k2 > 12 {
            wide_skips += 1;
            continue;
        }
        let e = shapes.entry((md.k, md.n, md.k2)).or_default();
        if e.len() < 2 { e.push(md); }
    }
    if frac_skips > 0 {
        println!("    PSK persistent split-K: {frac_skips} fractional chain shape(s) skipped (PSK bodies are integer 2..=6; serving falls back to the chain — the B6 host guard)");
    }
    if wide_skips > 0 {
        println!("    PSK persistent split-K: {wide_skips} wide chain shape(s) skipped (k2 > 12, e.g. the 8-bit MTP head: no persistent body — serve falls back to the plain chain via the fx width guard)");
    }
    if shapes.is_empty() {
        bail!("EXL3-KS-P FAIL: no split-K chain shapes in the pack (harness failure, nothing was tested)");
    }
    println!("    PSK persistent split-K: {} chain shape(s); slots single {slots1} / pair {slots2}", shapes.len());
    let word = |bits: usize, ks: u32, pf: i32| -> i32 { (bits as i32) | ((ks as i32) << 8) | (pf << 24) };
    let poison = f32::from_bits(0x7FC0_DEAD);
    let rd = |md: &ModuleMeta| -> Result<Vec<u16>> {
        Ok(md.trellis.read_bytes(&pack.dir)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    };
    let glist = |items: u32, slots: u32| -> Vec<u32> {
        let mut v = vec![1, 5, 47, 48, 96, items.min(slots), items.saturating_sub(1).max(1), items, items + 7];
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut cases = 0usize;
    for (&(k, n, bits), mods) in &shapes {
        let ks = crate::exl3_forward::chain_ks(k as i32, n as i32);
        let tiles = (n / 128) as u32;
        let items = tiles * ks;
        let d_tr0 = dev.htod_sync_copy(&rd(mods[0])?)?;
        let d_tr1 = dev.htod_sync_copy(&rd(mods[mods.len() - 1])?)?;
        let g1 = glist(items, slots1);
        let g2 = glist(2 * items, slots2);
        // (G, pf) combos: every G at the default depth, plus the default G at depths 0 and 2
        let combos = |gs: &[u32], gdef: u32| -> Vec<(u32, i32)> {
            let mut v: Vec<(u32, i32)> = gs.iter().map(|&g| (g, 1)).collect();
            v.push((gdef, 0));
            v.push((gdef, 2));
            v
        };
        let c1 = combos(&g1, items.min(slots1));
        let c2 = combos(&g2, (2 * items).min(slots2));
        let mut row0: Option<Vec<u16>> = None;
        for m in 1..=16usize {
            let need = ks as usize * m * n;
            let pz = vec![poison; need];
            let d_x0 = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?;           // row 0 shared by every m
            let d_x1 = dev.htod_sync_copy(&synth_x(m, k, 0xF00D ^ m as u32))?;
            let mut r0 = dev.htod_sync_copy(&pz)?;
            let mut p0 = dev.htod_sync_copy(&pz)?;
            let mut p1 = dev.htod_sync_copy(&pz)?;
            let mut w0 = dev.htod_sync_copy(&pz)?;
            let mut w1 = dev.htod_sync_copy(&pz)?;
            let old_cfg = |z: u32| LaunchConfig { grid_dim: (tiles, ks, z), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let new_cfg = |g: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f_old.clone().launch(old_cfg(1), (&d_tr0, &d_x0, &mut r0, m as i32, k as i32, n as i32, bits as i32))?; // k2-keyed (true k2 since B8)
                f_old2.clone().launch(old_cfg(2), (&d_tr0, &d_tr1, &d_x0, &d_x1, &mut p0, &mut p1,
                                                   m as i32, k as i32, n as i32, bits as i32))?; // k2-keyed
            }
            dev.synchronize()?;
            let ref0 = dev.dtoh_sync_copy(&r0)?;
            let (refp0, refp1) = (dev.dtoh_sync_copy(&p0)?, dev.dtoh_sync_copy(&p1)?);
            // the success signal: a finite, mostly-nonzero reference (and the old pair's member 0
            // IS the old single on the same inputs)
            for (tag, v) in [("single", &ref0), ("pair[0]", &refp0), ("pair[1]", &refp1)] {
                let live = v.iter().filter(|x| x.is_finite() && **x != 0.0).count();
                if live < need / 2 {
                    bail!("EXL3-KS-P FAIL: implausible {tag} reference K={k} N={n} b{bits} m={m} ({live}/{need} finite nonzero)");
                }
            }
            if ref0.iter().zip(refp0.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
                bail!("EXL3-KS-P FAIL: old pair member 0 != old single (K={k} N={n} m={m}) — reference inconsistent");
            }
            for &(g, pf) in &c1 {
                dev.htod_sync_copy_into(&pz, &mut w0)?;
                unsafe {
                    f_new.clone().launch(new_cfg(g), (&d_tr0, &d_x0, &mut w0, m as i32, k as i32, n as i32,
                                                      word(bits / 2, ks, pf), &d_tr0))?; // cfg is bits-keyed; loop var is k2
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&w0)?;
                let mism = got.iter().zip(ref0.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if mism > 0 {
                    let i = got.iter().zip(ref0.iter()).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
                    bail!("EXL3-KS-P FAIL: single K={k} N={n} b{bits} ks={ks} m={m} G={g} pf={pf}: {mism} of {need} partials differ \
                           (first [{i}] psk {:e} vs old {:e})", got[i], ref0[i]);
                }
                cases += 1;
            }
            for &(g, pf) in &c2 {
                dev.htod_sync_copy_into(&pz, &mut w0)?;
                dev.htod_sync_copy_into(&pz, &mut w1)?;
                unsafe {
                    f_new2.clone().launch(new_cfg(g), (&d_tr0, &d_tr1, &d_x0, &d_x1, &mut w0, &mut w1,
                                                       m as i32, k as i32, n as i32, word(bits / 2, ks, pf), &d_tr0, &d_tr1))?; // cfg is bits-keyed
                }
                dev.synchronize()?;
                let (a, b) = (dev.dtoh_sync_copy(&w0)?, dev.dtoh_sync_copy(&w1)?);
                let mism = a.iter().zip(refp0.iter()).chain(b.iter().zip(refp1.iter()))
                    .filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                if mism > 0 {
                    bail!("EXL3-KS-P FAIL: pair K={k} N={n} b{bits} ks={ks} m={m} G={g} pf={pf}: {mism} of {} partials differ",
                          2 * need);
                }
                cases += 1;
            }
            // y_raw through the unchanged combine on the PSK partials (w0 = single, default G):
            // row 0 bitwise across widths.
            dev.htod_sync_copy_into(&pz, &mut w0)?;
            let mut yr = dev.htod_sync_copy(&vec![0u16; m * n])?;
            unsafe {
                f_new.clone().launch(new_cfg(items.min(slots1)), (&d_tr0, &d_x0, &mut w0, m as i32, k as i32, n as i32,
                                                                  word(bits / 2, ks, 1), &d_tr0))?; // cfg is bits-keyed
                f_comb.clone().launch(LaunchConfig { grid_dim: (((m * n + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1),
                                                     shared_mem_bytes: 0 },
                                      (&w0, &mut yr, m as i32, n as i32, ks as i32))?;
            }
            dev.synchronize()?;
            let y = dev.dtoh_sync_copy(&yr)?;
            match &row0 {
                None => row0 = Some(y[..n].to_vec()),
                Some(r) => {
                    let mism = r.iter().zip(y[..n].iter()).filter(|(a, b)| a != b).count();
                    if mism > 0 { bail!("EXL3-KS-P BINV FAIL: K={k} N={n} b{bits}: y_raw row 0 at m={m} differs from m=1 in {mism} of {n}"); }
                }
            }
        }
        println!("      K={k:<5} N={n:<6} b{bits} ks={ks:<3} items {items:>4}: bitwise == old grid, M 1..16, \
                  single G {g1:?} + pair G {g2:?}, pf 0/1/2; y_raw row-0 binv PASS ({})",
                 mods.iter().map(|md| md.name.as_str()).collect::<Vec<_>>().join(" | "));
    }
    println!("    EXL3-KS-P: PASS ({cases} (shape, M, G, pf) launches bitwise == exl3_hmma_gemm_ks(_x2))");
    Ok(())
}

// ---------------------------------------------------------------------------
// WP21 (split-K fixup epilogue, the served default): exl3_hmma_gemm_ks_fx and exl3_hmma_gemm_ks_fx_x2
// (the x2 through the SERVED raw launcher exl3_forward::wp21_fx_x2_raw) must be BITWISE equal to the
// 3-launch tail they replace — exl3_had_suh -> exl3_hmma_gemm_ks -> exl3_ks_combine -> exl3_had_svh
// (dense_ref) — on every y element AND every fp32 partial ws[slab][row][col] (the fused body stores
// the same partials), for every split chain shape of the pack (non-expert, non-head, chain_ks > 1),
// M = 1..16, single and same-shape pair (two distinct modules when the pack has them; each member
// with its own counter slice). Every fused launch runs TWICE on the same counters with partials
// NaN-poisoned and y 0xFFFF-poisoned before each run: the second run only writes y if the first left
// every tile counter re-armed at 0 (graph-replay safety); counters are read back at 0 after each
// run. y row 0 is bitwise across M (batch invariance). REPRIME's wp22r1 form (M = 17 / 28 / 32 as
// <= 16-row slabs sharing ws and the counters, each fused launch writing ITS rows of y) must equal
// the old slab loop (ks GEMM + combine per slab, one svh over all rows), and its row 0 must equal the
// M <= 16 chains' row 0. The reference must be plausible (finite, mostly nonzero).
// ---------------------------------------------------------------------------
pub(super) fn probe_wp21(dev: &std::sync::Arc<CudaDevice>, pack: &Exl3Pack) -> Result<()> {
    use std::collections::BTreeMap;
    let st = fork_blocking_stream(dev);
    let gf = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let f_fx = gf("exl3_hmma_gemm_ks_fx")?;
    let rd16 = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let mut shapes: BTreeMap<(usize, usize, usize), Vec<&ModuleMeta>> = BTreeMap::new();
    for md in &pack.modules {
        if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") { continue; }
        if md.k % 128 != 0 || md.n % 128 != 0 || !(3..=7).contains(&(md.k2 / 2)) { continue; }
        let ks = crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize;
        if ks <= 1 || (md.k / 16) % ks != 0 { continue; }
        let e = shapes.entry((md.k, md.n, md.k2 / 2)).or_default();
        if e.len() < 2 { e.push(md); }
    }
    if shapes.is_empty() {
        bail!("EXL3-WP21 FAIL: no split-K chain shapes in the pack (harness failure, nothing was tested)");
    }
    let load = |md: &ModuleMeta| -> Result<DenseMod> {
        Ok(DenseMod {
            name: md.name.clone(),
            tr: dev.htod_sync_copy(&rd16(md.trellis.read_bytes(&pack.dir)?))?,
            suh: dev.htod_sync_copy(&le_f16_bits(&md.suh.read_bytes(&pack.dir)?))?,
            svh: dev.htod_sync_copy(&le_f16_bits(&md.svh.read_bytes(&pack.dir)?))?,
            k: md.k, n: md.n, bits: md.k2 / 2,
            ks: crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize,
        })
    };
    let cfg = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    let poison = f32::from_bits(0x7FC0_DEAD);
    let nz = |c: &CudaSlice<u32>| -> Result<usize> { Ok(dev.dtoh_sync_copy(c)?.iter().filter(|&&v| v != 0).count()) };
    let pmism = |a: &[f32], b: &[f32]| a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let ymism = |a: &[u16], b: &[u16]| a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    println!("    WP21 split-K fixup epilogue: {} chain shape(s), fused vs 3-launch tail (single + pair + wp22r1 slabs)",
             shapes.len());
    let (mut launches, mut slab_cells) = (0usize, 0usize);
    for (&(k, n, bits), mods) in &shapes {
        let a = load(mods[0])?;
        let b = load(mods[mods.len() - 1])?;
        let ks = a.ks;
        let tiles = n / 128;
        let cnt0 = dev.htod_sync_copy(&vec![0u32; tiles.max(1)])?;
        let cnt1 = dev.htod_sync_copy(&vec![0u32; tiles.max(1)])?;
        let mut row0: Option<Vec<u16>> = None;
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?; // row 0 shared by every m
            let (rp0, ry0) = dense_ref(dev, &st, &a, &x, None, m)?;
            let (rp1, ry1) = dense_ref(dev, &st, &b, &x, None, m)?;
            for (tag, p, y) in [("member 0", &rp0, &ry0), ("member 1", &rp1, &ry1)] {
                let pl = p.iter().filter(|v| v.is_finite() && **v != 0.0).count();
                let yl = y.iter().filter(|&&v| v != 0 && f16::from_bits(v).to_f32().is_finite()).count();
                if pl < p.len() / 2 || yl < y.len() / 2 {
                    bail!("EXL3-WP21 FAIL: implausible {tag} reference K={k} N={n} b{bits} m={m} ({pl}/{} partials, {yl}/{} y live)",
                          p.len(), y.len());
                }
            }
            let mut xh0 = dev.htod_sync_copy(&vec![0u16; m * k])?;
            let mut xh1 = dev.htod_sync_copy(&vec![0u16; m * k])?;
            unsafe {
                gf("exl3_had_suh_x2")?.launch_on_stream(&st, cfg(((m * k / 128) as u32, 1, 2), 32),
                                                        (&x, &a.suh, &b.suh, &mut xh0, &mut xh1, k as i32))?;
            }
            let mut ws0 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
            let mut ws1 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
            let mut y0 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            let mut y1 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            for rep in 0..2 {
                // single (member 0)
                dev.htod_sync_copy_into(&vec![poison; ks * m * n], &mut ws0)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut y0)?;
                unsafe {
                    f_fx.clone().launch_on_stream(&st, cfg((tiles as u32, ks as u32, 1), 256),
                        (&a.tr, &xh0, &mut ws0, &a.svh, &mut y0, &cnt0, m as i32, k as i32, n as i32, (2 * bits) as i32))?;
                }
                dev.synchronize()?;
                launches += 1;
                let (p, y, c) = (dev.dtoh_sync_copy(&ws0)?, dev.dtoh_sync_copy(&y0)?, nz(&cnt0)?);
                let (pm, ym) = (pmism(&p, &rp0), ymism(&y, &ry0));
                if pm + ym + c > 0 {
                    bail!("EXL3-WP21 FAIL: single K={k} N={n} b{bits} ks={ks} m={m} run {rep}: partials mism {pm}, y mism {ym} \
                           of {}, counters not re-armed {c} ({})", m * n, a.name);
                }
                if rep == 0 {
                    match &row0 {
                        None => row0 = Some(y[..n].to_vec()),
                        Some(r) => {
                            let d = ymism(r, &y[..n]);
                            if d > 0 { bail!("EXL3-WP21 BINV FAIL: K={k} N={n} b{bits}: y row 0 at m={m} differs from m=1 in {d} of {n}"); }
                        }
                    }
                }
                // pair (members 0 and 1, same x, own counter slices) — the served raw launcher
                dev.htod_sync_copy_into(&vec![poison; ks * m * n], &mut ws0)?;
                dev.htod_sync_copy_into(&vec![poison; ks * m * n], &mut ws1)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut y0)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut y1)?;
                let pp: [u64; 12] = [
                    *a.tr.device_ptr(), *b.tr.device_ptr(), *xh0.device_ptr(), *xh1.device_ptr(),
                    *ws0.device_ptr(), *ws1.device_ptr(), *a.svh.device_ptr(), *b.svh.device_ptr(),
                    *y0.device_ptr(), *y1.device_ptr(), *cnt0.device_ptr(), *cnt1.device_ptr(),
                ];
                crate::exl3_forward::wp21_fx_x2_raw(st.stream, pp, m, k as i32, n as i32, (2 * bits) as i32, ks as u32)?; // k2, as the served caller passes
                dev.synchronize()?;
                launches += 1;
                let (p0, p1) = (dev.dtoh_sync_copy(&ws0)?, dev.dtoh_sync_copy(&ws1)?);
                let (g0, g1) = (dev.dtoh_sync_copy(&y0)?, dev.dtoh_sync_copy(&y1)?);
                let c = nz(&cnt0)? + nz(&cnt1)?;
                let (pm, ym) = (pmism(&p0, &rp0) + pmism(&p1, &rp1), ymism(&g0, &ry0) + ymism(&g1, &ry1));
                if pm + ym + c > 0 {
                    bail!("EXL3-WP21 FAIL: pair K={k} N={n} b{bits} ks={ks} m={m} run {rep}: partials mism {pm}, y mism {ym} \
                           of {}, counters not re-armed {c} ({} | {})", 2 * m * n, a.name, b.name);
                }
            }
        }
        // REPRIME wp22r1 row slabs (m > 16): fused per slab vs the old slab loop, sharing ws + counters.
        let r0_ref = row0.clone().unwrap_or_default();
        for m in [17usize, 28, 32] {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?;
            let mut xh = dev.htod_sync_copy(&vec![0u16; m * k])?;
            let mut ws = dev.htod_sync_copy(&vec![poison; ks * 16 * n])?;
            let yr = dev.htod_sync_copy(&vec![0u16; m * n])?; // written through yr_p (slab row offsets)
            let mut yref = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            let mut yfx = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            let (xh_p, yr_p, yf_p) = (*xh.device_ptr(), *yr.device_ptr(), *yfx.device_ptr());
            let slabs: Vec<(usize, usize)> = (0..m).step_by(16).map(|r0| (r0, (m - r0).min(16))).collect();
            unsafe {
                gf("exl3_had_suh")?.launch_on_stream(&st, cfg(((m * k / 128) as u32, 1, 1), 32), (&x, &a.suh, &mut xh, k as i32))?;
                for &(r0, ms) in &slabs {
                    gf("exl3_hmma_gemm_ks")?.launch_on_stream(&st, cfg((tiles as u32, ks as u32, 1), 256),
                        (&a.tr, xh_p + (r0 * k * 2) as u64, &mut ws, ms as i32, k as i32, n as i32, (2 * bits) as i32))?;
                    gf("exl3_ks_combine")?.launch_on_stream(&st, cfg((((ms * n + 255) / 256) as u32, 1, 1), 256),
                        (&ws, yr_p + (r0 * n * 2) as u64, ms as i32, n as i32, ks as i32))?;
                }
                gf("exl3_had_svh")?.launch_on_stream(&st, cfg(((m * n / 128) as u32, 1, 1), 32), (&yr, &a.svh, &mut yref, n as i32))?;
            }
            dev.synchronize()?;
            let want = dev.dtoh_sync_copy(&yref)?;
            let live = want.iter().filter(|&&v| v != 0 && v != 0xFFFF).count();
            if live < want.len() / 2 {
                bail!("EXL3-WP21 FAIL: implausible slab reference K={k} N={n} b{bits} m={m} ({live}/{} live)", want.len());
            }
            for rep in 0..2 {
                dev.htod_sync_copy_into(&vec![poison; ks * 16 * n], &mut ws)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut yfx)?;
                for &(r0, ms) in &slabs {
                    unsafe {
                        f_fx.clone().launch_on_stream(&st, cfg((tiles as u32, ks as u32, 1), 256),
                            (&a.tr, xh_p + (r0 * k * 2) as u64, &mut ws, &a.svh, yf_p + (r0 * n * 2) as u64, &cnt0,
                             ms as i32, k as i32, n as i32, (2 * bits) as i32))?;
                    }
                    launches += 1;
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&yfx)?;
                let (ym, c) = (ymism(&got, &want), nz(&cnt0)?);
                let r0d = if r0_ref.len() == n { ymism(&got[..n], &r0_ref) } else { n };
                if ym + c + r0d > 0 {
                    bail!("EXL3-WP21 FAIL: wp22r1 slabs K={k} N={n} b{bits} ks={ks} m={m} ({} slabs) run {rep}: y mism {ym} of {}, \
                           counters not re-armed {c}, row 0 vs the m<=16 chains {r0d} ({})", slabs.len(), m * n, a.name);
                }
            }
            slab_cells += 1;
        }
        println!("      K={k:<5} N={n:<6} b{bits} ks={ks:<3} tiles {tiles:>3}: fused == 3-launch (partials + y), M 1..16 single + pair, \
                  2 runs/cell (counters re-armed), y row-0 binv; wp22r1 slabs M 17/28/32 PASS ({})",
                 mods.iter().map(|md| md.name.as_str()).collect::<Vec<_>>().join(" | "));
    }
    println!("    EXL3-WP21: PASS ({launches} fused launches bitwise == exl3_hmma_gemm_ks(_x2) + exl3_ks_combine(_x2) + \
              exl3_had_svh(_x2); {} shape(s), {slab_cells} wp22r1 slab cell(s))", shapes.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// W4/DENSE gate: exl3_dks_* — launched through the SERVED launcher (exl3_forward::dense::
// probe_launch -> dks_launch) — must be BITWISE equal to the old chain it replaces,
//   [xq_silu_mul ->] exl3_had_suh -> exl3_hmma_gemm_ks -> exl3_ks_combine -> exl3_had_svh,
// member by member: every fp32 partial ws[slab][row][col] and every y element, for every split
// chain shape of the pack (singles), same-shape pairs, one same-K multi-member group (big + small
// members in one item list), the silu prologue (shared-expert down), M = 1..16, grid sizes G (the
// served rule, 1, odd, 40, 48, more CTAs than items), ring depths, and the escape flag sets
// (partials-only tail, staged xh). Partials are NaN-poisoned and y 0xFFFF-poisoned before every
// launch (an unwritten element cannot pass); every fixup launch must leave its tile counters at 0
// (self re-arming); the reference must be plausible; y row 0 is bitwise across M (batch invariance).
// A (job, M) cell the served grid rule cannot place (dks_plan: no G fits the A-entry / smem budget,
// e.g. the K2560 x N10240 / N12288 same-shape pairs at M 9..16) is SKIPPED with a printed line — the
// served path falls back to the old chain there (dense_chains -> Ok(false), nothing launched). For
// pairs that fallback is exl3_chain_pair's x2 chain, which is run here and bit-checked against the
// old chains; for singles / multi groups / the silu down it is the reference chain itself. A skip
// never ends the probe (the later binv sections must still run); zero DENSE launches is a FAIL.
// ---------------------------------------------------------------------------
struct DenseMod {
    name: String,
    tr: CudaSlice<u16>,
    suh: CudaSlice<u16>,
    svh: CudaSlice<u16>,
    k: usize,
    n: usize,
    bits: usize,
    ks: usize,
}

/// The old chain on `x` (m rows) for one member -> (fp32 partials, y).
fn dense_ref(dev: &std::sync::Arc<CudaDevice>, st: &CudaStream, md: &DenseMod, x: &CudaSlice<u16>,
             x2: Option<&CudaSlice<u16>>, m: usize) -> Result<(Vec<f32>, Vec<u16>)> {
    let gf = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (k, n, ks) = (md.k, md.n, md.ks);
    let mut xh = dev.htod_sync_copy(&vec![0u16; m * k])?;
    let mut ws = dev.htod_sync_copy(&vec![0f32; ks * m * n])?;
    let mut yr = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut y = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let cfg = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        let xin = match x2 {
            Some(u) => {
                let mut s = dev.htod_sync_copy(&vec![0u16; m * k])?;
                gf("xq_silu_mul")?.launch_on_stream(st, cfg((((m * k) as u32) / 256 + 1, 1, 1), 256),
                                                    (&mut s, x, u, (m * k) as i64))?;
                Some(s)
            }
            None => None,
        };
        let xr = xin.as_ref().unwrap_or(x);
        gf("exl3_had_suh")?.launch_on_stream(st, cfg(((m * k / 128) as u32, 1, 1), 32), (xr, &md.suh, &mut xh, k as i32))?;
        gf("exl3_hmma_gemm_ks")?.launch_on_stream(st, cfg(((n / 128) as u32, ks as u32, 1), 256),
                                                  (&md.tr, &xh, &mut ws, m as i32, k as i32, n as i32, (2 * md.bits) as i32))?;
        gf("exl3_ks_combine")?.launch_on_stream(st, cfg((((m * n + 255) / 256) as u32, 1, 1), 256),
                                                (&ws, &mut yr, m as i32, n as i32, ks as i32))?;
        gf("exl3_had_svh")?.launch_on_stream(st, cfg(((m * n / 128) as u32, 1, 1), 32), (&yr, &md.svh, &mut y, n as i32))?;
        dev.synchronize()?;
        drop(xin);
    }
    Ok((dev.dtoh_sync_copy(&ws)?, dev.dtoh_sync_copy(&y)?))
}

/// The SERVED fallback of a same-shape pair the DENSE grid rule cannot place: exl3_chain_pair past
/// its DENSE block with PSK at its default (off) — exl3_had_suh_x2 -> exl3_hmma_gemm_ks_x2 ->
/// exl3_ks_combine_x2 -> exl3_had_svh_x2 on x (m rows) -> per member (fp32 partials, y). Partials
/// NaN-poisoned, y 0xFFFF-poisoned first (an unwritten element cannot compare equal).
fn dense_pair_fallback(dev: &std::sync::Arc<CudaDevice>, st: &CudaStream, a: &DenseMod, b: &DenseMod,
                       x: &CudaSlice<u16>, m: usize) -> Result<[(Vec<f32>, Vec<u16>); 2]> {
    let gf = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (k, n, ks) = (a.k, a.n, a.ks);
    anyhow::ensure!(b.k == k && b.n == n && b.ks == ks && b.bits == a.bits,
                    "pair fallback: members differ (K{k} N{n} ks{ks} b{} vs K{} N{} ks{} b{})", a.bits, b.k, b.n, b.ks, b.bits);
    let poison = f32::from_bits(0x7FC0_DEAD);
    let mut xh0 = dev.htod_sync_copy(&vec![0u16; m * k])?;
    let mut xh1 = dev.htod_sync_copy(&vec![0u16; m * k])?;
    let mut ws0 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
    let mut ws1 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
    let mut yr0 = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut yr1 = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut y0 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
    let mut y1 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
    let cfg = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        gf("exl3_had_suh_x2")?.launch_on_stream(st, cfg(((m * k / 128) as u32, 1, 2), 32),
                                                (x, &a.suh, &b.suh, &mut xh0, &mut xh1, k as i32))?;
        gf("exl3_hmma_gemm_ks_x2")?.launch_on_stream(st, cfg(((n / 128) as u32, ks as u32, 2), 256),
                                                     (&a.tr, &b.tr, &xh0, &xh1, &mut ws0, &mut ws1,
                                                      m as i32, k as i32, n as i32, (2 * a.bits) as i32))?;
        gf("exl3_ks_combine_x2")?.launch_on_stream(st, cfg((((m * n + 255) / 256) as u32, 1, 2), 256),
                                                   (&ws0, &ws1, &mut yr0, &mut yr1, m as i32, n as i32, ks as i32))?;
        gf("exl3_had_svh_x2")?.launch_on_stream(st, cfg(((m * n / 128) as u32, 1, 2), 32),
                                                (&yr0, &yr1, &a.svh, &b.svh, &mut y0, &mut y1, n as i32))?;
    }
    dev.synchronize()?;
    Ok([(dev.dtoh_sync_copy(&ws0)?, dev.dtoh_sync_copy(&y0)?), (dev.dtoh_sync_copy(&ws1)?, dev.dtoh_sync_copy(&y1)?)])
}

/// Widths as compact ranges ("1..8, 11, 14..16"; "none" when empty).
fn m_ranges(ms: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ms.len() {
        let mut j = i;
        while j + 1 < ms.len() && ms[j + 1] == ms[j] + 1 { j += 1; }
        out.push(if j > i { format!("{}..{}", ms[i], ms[j]) } else { ms[i].to_string() });
        i = j + 1;
    }
    if out.is_empty() { "none".to_string() } else { out.join(", ") }
}

pub(super) fn probe_dense(dev: &std::sync::Arc<CudaDevice>, pack: &Exl3Pack) -> Result<()> {
    use crate::exl3_forward::dense::{self as dn, ProbeMem, DKS_F_FIX, DKS_F_SILU, DKS_F_SUH};
    use std::collections::BTreeMap;
    let st = fork_blocking_stream(dev);
    let rd16 = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    // every split chain module (non-expert, non-head), grouped by (K, N, bits) -> up to 2 modules
    let mut shapes: BTreeMap<(usize, usize, usize), Vec<&ModuleMeta>> = BTreeMap::new();
    for md in &pack.modules {
        if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") { continue; }
        let ks = crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize;
        if !dn::dks_member_ok(md.k, md.n, ks) || !(3..=5).contains(&(md.k2 / 2)) { continue; }
        let e = shapes.entry((md.k, md.n, md.k2 / 2)).or_default();
        if e.len() < 2 { e.push(md); }
    }
    if shapes.is_empty() {
        // Gate semantics: a pack whose split-chain modules are ALL outside the probed width
        // domain is SKIPPED with a visible reason (the chains still run — WP21/KS-P cover them);
        // only a pack with no chain-shaped modules at all is a harness failure.
        let mut chain_total = 0usize;
        let mut chain_wide = 0usize;
        for md in &pack.modules {
            if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") { continue; }
            let ks = crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize;
            if !dn::dks_member_ok(md.k, md.n, ks) || ks <= 1 { continue; }
            chain_total += 1;
            if !(3..=5).contains(&(md.k2 / 2)) { chain_wide += 1; }
        }
        if chain_total > 0 && chain_wide == chain_total {
            println!("    EXL3-DENSE: SKIP ({chain_total} chain shape(s) all outside the probed widths 3..=5 (k2 12/14/16); the split chains themselves are covered by EXL3-KS-P and EXL3-WP21)");
            return Ok(());
        }
        bail!("EXL3-DENSE FAIL: no split chain shapes in the pack (harness failure, nothing was tested)");
    }
    let load = |md: &ModuleMeta| -> Result<DenseMod> {
        Ok(DenseMod {
            name: md.name.clone(),
            tr: dev.htod_sync_copy(&rd16(md.trellis.read_bytes(&pack.dir)?))?,
            suh: dev.htod_sync_copy(&le_f16_bits(&md.suh.read_bytes(&pack.dir)?))?,
            svh: dev.htod_sync_copy(&le_f16_bits(&md.svh.read_bytes(&pack.dir)?))?,
            k: md.k, n: md.n, bits: md.k2 / 2,
            ks: crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize,
        })
    };
    let poison = f32::from_bits(0x7FC0_DEAD);
    let served_nst = dn::dense_nst();
    let mut launches = 0usize;

    // One job: members `mods` (same K / bits) on x (+x2) at width m, grid g, ring nst, flags; checks
    // every member's partials (+ y with FIX) against `refs`, counters re-armed. Returns member 0's y.
    #[allow(clippy::too_many_arguments)]
    let mut run = |mods: &[&DenseMod], refs: &[(Vec<f32>, Vec<u16>)], x: &CudaSlice<u16>, x2: Option<&CudaSlice<u16>>,
                   m: usize, g: usize, nst: usize, flags: i32, tag: &str| -> Result<Vec<u16>> {
        let k = mods[0].k;
        let mut wss = Vec::new();
        let mut ys = Vec::new();
        let mut cnts = Vec::new();
        let mut xhs = Vec::new();
        let mut pm = Vec::new();
        for md in mods {
            wss.push(dev.htod_sync_copy(&vec![poison; md.ks * m * md.n])?);
            ys.push(dev.htod_sync_copy(&vec![0xFFFFu16; m * md.n])?);
            cnts.push(dev.htod_sync_copy(&vec![0u32; (md.n / 128).max(1)])?);
            // staged xh (flags without SUH): the old had_suh on the same x
            let mut xh = dev.htod_sync_copy(&vec![0u16; m * k])?;
            if flags & DKS_F_SUH == 0 {
                unsafe {
                    dev.get_func(MODULE, "exl3_had_suh").context("exl3_had_suh")?.launch_on_stream(&st,
                        LaunchConfig { grid_dim: ((m * k / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                        (x, &md.suh, &mut xh, k as i32))?;
                }
            }
            xhs.push(xh);
        }
        for (j, md) in mods.iter().enumerate() {
            pm.push(ProbeMem { tr: *md.tr.device_ptr(), suh: *md.suh.device_ptr(), svh: *md.svh.device_ptr(),
                               xh: *xhs[j].device_ptr(), ws: *wss[j].device_ptr(), y: *ys[j].device_ptr(),
                               cnt: *cnts[j].device_ptr(), n: md.n, ks: md.ks });
        }
        dev.synchronize()?;
        let x2p = x2.map_or(0u64, |u| *u.device_ptr());
        if !dn::probe_launch(st.stream, &pm, *x.device_ptr(), x2p, m, k, mods[0].bits, g, nst, flags)? {
            bail!("EXL3-DENSE FAIL [{tag}]: launcher refused m={m} G={g} nst={nst} flags={flags}");
        }
        dev.synchronize()?;
        launches += 1;
        let mut y0 = Vec::new();
        for (j, md) in mods.iter().enumerate() {
            let p = dev.dtoh_sync_copy(&wss[j])?;
            let mism = p.iter().zip(refs[j].0.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if mism > 0 {
                let i = p.iter().zip(refs[j].0.iter()).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
                bail!("EXL3-DENSE FAIL [{tag}] member {j} ({}) K={k} N={} ks={} m={m} G={g} nst={nst} flags={flags}: \
                       {mism} of {} partials differ (first [{i}] new {:e} old {:e})",
                      md.name, md.n, md.ks, p.len(), p[i], refs[j].0[i]);
            }
            if flags & DKS_F_FIX != 0 {
                let y = dev.dtoh_sync_copy(&ys[j])?;
                let mism = y.iter().zip(refs[j].1.iter()).filter(|(a, b)| a != b).count();
                if mism > 0 {
                    let i = y.iter().zip(refs[j].1.iter()).position(|(a, b)| a != b).unwrap_or(0);
                    bail!("EXL3-DENSE FAIL [{tag}] member {j} ({}) K={k} N={} m={m} G={g} nst={nst} flags={flags}: \
                           {mism} of {} y differ (first (row {}, col {}) new {:04x} old {:04x})",
                          md.name, md.n, y.len(), i / md.n, i % md.n, y[i], refs[j].1[i]);
                }
                let nz = dev.dtoh_sync_copy(&cnts[j])?.iter().filter(|&&c| c != 0).count();
                if nz > 0 {
                    bail!("EXL3-DENSE FAIL [{tag}] member {j} K={k} N={} m={m} G={g}: {nz} tile counters not re-armed", md.n);
                }
                if j == 0 { y0 = y; }
            }
        }
        Ok(y0)
    };
    let plausible = |r: &(Vec<f32>, Vec<u16>), tag: &str, m: usize| -> Result<()> {
        let live = r.0.iter().filter(|v| v.is_finite() && **v != 0.0).count();
        let ylive = r.1.iter().filter(|v| **v != 0 && **v != 0xFFFF).count();
        if live < r.0.len() / 2 || ylive < r.1.len() / 2 {
            bail!("EXL3-DENSE FAIL [{tag}]: implausible reference at m={m} ({live}/{} partials, {ylive}/{} y live)",
                  r.0.len(), r.1.len());
        }
        Ok(())
    };
    let full = DKS_F_SUH | DKS_F_FIX;
    println!("    W4/DENSE: {} split chain shape(s); served ring nst {served_nst}", shapes.len());
    // (job, m) cells with no feasible served grid (skipped), and pair fallbacks bit-checked there
    let (mut nskip, mut nfb) = (0usize, 0usize);
    const NOGRID: &str = "no feasible grid (served path falls back to the old chain) — skipped";

    // ---- singles: every shape, M 1..16, a G sweep; ring depths + escape flag sets at M 1/8/16
    let mut mods_all: Vec<DenseMod> = Vec::new();
    let mut idx_of: BTreeMap<(usize, usize, usize), Vec<usize>> = BTreeMap::new();
    for (&key, mds) in &shapes {
        for md in mds {
            idx_of.entry(key).or_default().push(mods_all.len());
            mods_all.push(load(md)?);
        }
    }
    for (&(k, n, bits), ids) in &idx_of {
        let md = &mods_all[ids[0]];
        let items = (n / 128) * md.ks;
        let mut row0: Option<(usize, Vec<u16>)> = None;
        let mut gl_all = Vec::new();
        let (mut ran, mut skipped) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?;
            let r = dense_ref(dev, &st, md, &x, None, m)?;
            plausible(&r, "single", m)?;
            let Some(gs) = dn::probe_grid(dev, k, bits, &[(n, md.ks)], m) else {
                // served: exl3_chain_impl's own path — the reference chain above
                println!("      EXL3-DENSE single K={k} N={n} b{bits} m={m}: {NOGRID} (the fallback is this section's reference chain)");
                skipped.push(m);
                nskip += 1;
                continue;
            };
            let mut gl = vec![gs, 1, 7, 13, 40, 48, items + 3];
            gl.sort_unstable();
            gl.dedup();
            // only grids the launcher accepts (A entries / smem budget) — the served picker and the
            // --w4dense-g override never choose another
            gl.retain(|&g| dn::dks_feasible(&[(n, md.ks)], k, bits, served_nst, g, m));
            for &g in &gl {
                let y = run(&[md], std::slice::from_ref(&r), &x, None, m, g, served_nst, full, "single")?;
                if g == gs {
                    match &row0 {
                        None => row0 = Some((m, y[..n].to_vec())),
                        Some((m0, r0)) => if r0[..] != y[..n] {
                            bail!("EXL3-DENSE BINV FAIL: K={k} N={n} b{bits}: y row 0 at m={m} differs from m={m0}");
                        },
                    }
                }
            }
            if matches!(m, 1 | 8 | 16) {
                let nsts: &[usize] = if bits == 3 { &[6] } else { &[4, 6, 8] };
                for &ns in nsts {
                    if dn::dks_feasible(&[(n, md.ks)], k, bits, ns, gs, m) {
                        run(&[md], std::slice::from_ref(&r), &x, None, m, gs, ns, full, "single ring")?;
                    }
                }
                for fl in [DKS_F_FIX, DKS_F_SUH, 0] {
                    run(&[md], std::slice::from_ref(&r), &x, None, m, gs, served_nst, fl, "single flags")?;
                }
            }
            gl_all = gl;
            ran.push(m);
        }
        let sk = if skipped.is_empty() { String::new() } else { format!(" [M {} skipped: no feasible grid]", m_ranges(&skipped)) };
        if ran.is_empty() {
            println!("      K={k:<5} N={n:<6} b{bits} ks={:<3} items {items:>4}: single — every width skipped (no feasible grid; \
                      the served path runs the old chain) ({})", md.ks, md.name);
        } else {
            println!("      K={k:<5} N={n:<6} b{bits} ks={:<3} items {items:>4}: single bitwise == old chain, M {}, G {gl_all:?}, \
                      nst sweep + flag sets at M 1/8/16, counters re-armed, y row-0 binv PASS{sk} ({})",
                     md.ks, m_ranges(&ran), md.name);
        }
    }

    // ---- pairs: two modules of one shape on the same x (the k|v / gate|up launch shape)
    let mut pairs = 0usize;
    for (&(k, n, bits), ids) in &idx_of {
        if ids.len() < 2 { continue; }
        let (a, b) = (&mods_all[ids[0]], &mods_all[ids[1]]);
        let items = 2 * (n / 128) * a.ks;
        let (mut ran, mut fb) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0xA11C))?;
            let refs = [dense_ref(dev, &st, a, &x, None, m)?, dense_ref(dev, &st, b, &x, None, m)?];
            let Some(gs) = dn::probe_grid(dev, k, bits, &[(n, a.ks), (n, b.ks)], m) else {
                // served: exl3_chain_pair -> dense_chains declines (nothing launched) -> the x2 chain;
                // run it and bit-check both members against the old chains
                plausible(&refs[0], "pair fallback", m)?;
                plausible(&refs[1], "pair fallback", m)?;
                let got = dense_pair_fallback(dev, &st, a, b, &x, m)?;
                for (j, (g, r)) in got.iter().zip(refs.iter()).enumerate() {
                    let pm = g.0.iter().zip(r.0.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    let ym = g.1.iter().zip(r.1.iter()).filter(|(p, q)| p != q).count();
                    if pm + ym > 0 {
                        bail!("EXL3-DENSE FAIL [pair fallback] member {j} ({}) K={k} N={n} b{bits} m={m}: the served x2 fallback \
                               chain differs from the old chain ({pm} of {} partials, {ym} of {} y)",
                              if j == 0 { &a.name } else { &b.name }, g.0.len(), g.1.len());
                    }
                }
                println!("      EXL3-DENSE pair K={k} N={n} b{bits} m={m}: {NOGRID}; fallback x2 chain (exl3_had_suh_x2 + \
                          exl3_hmma_gemm_ks_x2 + exl3_ks_combine_x2 + exl3_had_svh_x2) bitwise == old chains (partials + y, \
                          both members)");
                fb.push(m);
                nskip += 1;
                nfb += 1;
                continue;
            };
            let mut gl = vec![gs, 1, 13, items + 3];
            gl.sort_unstable();
            gl.dedup();
            gl.retain(|&g| dn::dks_feasible(&[(n, a.ks), (n, b.ks)], k, bits, served_nst, g, m));
            for &g in &gl { run(&[a, b], &refs, &x, None, m, g, served_nst, full, "pair")?; }
            if matches!(m, 1 | 8 | 16) { run(&[a, b], &refs, &x, None, m, gs, served_nst, DKS_F_FIX, "pair staged xh")?; }
            ran.push(m);
        }
        pairs += 1;
        println!("      pair K={k} N={n} b{bits}: DENSE bitwise == old chains at M {}{} ({} | {})", m_ranges(&ran),
                 if fb.is_empty() { String::new() }
                 else { format!("; M {}: no feasible grid, served x2 fallback bitwise == old chains", m_ranges(&fb)) },
                 a.name, b.name);
    }

    // ---- multi-member group: per (K, bits), the largest-N module first, then the smallest shapes'
    // modules (the attention q + k|v + indexer composition when the pack has it), up to 4 members.
    let mut groups = 0usize;
    let mut by_kb: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (&(k, _n, bits), ids) in &idx_of { by_kb.entry((k, bits)).or_default().extend(ids.iter().copied()); }
    for (&(k, bits), ids) in &by_kb {
        if ids.len() < 3 { continue; }
        let mut sorted = ids.clone();
        sorted.sort_by_key(|&i| std::cmp::Reverse(mods_all[i].n));
        let mut pick = vec![sorted[0]];
        for &i in sorted.iter().rev() { if pick.len() < 4 && !pick.contains(&i) { pick.push(i); } }
        let mods: Vec<&DenseMod> = pick.iter().map(|&i| &mods_all[i]).collect();
        let mem: Vec<(usize, usize)> = mods.iter().map(|d| (d.n, d.ks)).collect();
        let (mut ran, mut skipped) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let Some(gs) = dn::probe_grid(dev, k, bits, &mem, m) else {
                // served: the group splits back to its own chains (each the reference chain here)
                println!("      EXL3-DENSE multi K={k} b{bits} {} m={m}: {NOGRID} (the fallback is this section's reference chains)",
                         mods.iter().map(|d| format!("N{}", d.n)).collect::<Vec<_>>().join("+"));
                skipped.push(m);
                nskip += 1;
                continue;
            };
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x6A0F))?;
            let refs: Vec<(Vec<f32>, Vec<u16>)> = mods.iter().map(|d| dense_ref(dev, &st, d, &x, None, m)).collect::<Result<_>>()?;
            let mut gl = vec![gs, 1, 13, 40, 48];
            gl.sort_unstable();
            gl.dedup();
            gl.retain(|&g| dn::dks_feasible(&mem, k, bits, served_nst, g, m));
            for &g in &gl { run(&mods, &refs, &x, None, m, g, served_nst, full, "multi")?; }
            ran.push(m);
        }
        groups += 1;
        println!("      multi K={k} b{bits} [{}]: bitwise == the old chains, M {}{}",
                 mods.iter().map(|d| format!("N{} {}", d.n, d.name)).collect::<Vec<_>>().join(" + "), m_ranges(&ran),
                 if skipped.is_empty() { String::new() } else { format!(" [M {} skipped: no feasible grid]", m_ranges(&skipped)) });
    }

    // ---- silu prologue (the shared-expert down): x := f16(silu(x) * x2) before the suh
    let silu_shape = idx_of.keys().find(|&&(k, _, _)| k == 640).or_else(|| idx_of.keys().next()).copied();
    if let Some(key) = silu_shape {
        let md = &mods_all[idx_of[&key][0]];
        let (k, n, bits) = key;
        let (mut ran, mut skipped) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x51A0))?;
            let u = dev.htod_sync_copy(&synth_x(m, k, 0x0B0E))?;
            let r = dense_ref(dev, &st, md, &x, Some(&u), m)?;
            plausible(&r, "silu", m)?;
            let Some(gs) = dn::probe_grid(dev, k, bits, &[(n, md.ks)], m) else {
                // served: shared_down's xq_silu_mul + exl3_chain — this section's reference chain
                println!("      EXL3-DENSE silu K={k} N={n} b{bits} m={m}: {NOGRID} (the fallback is this section's reference chain)");
                skipped.push(m);
                nskip += 1;
                continue;
            };
            for g in [gs, 1, 13] {
                if g == gs || dn::dks_feasible(&[(n, md.ks)], k, bits, served_nst, g, m) {
                    run(&[md], std::slice::from_ref(&r), &x, Some(&u), m, g, served_nst, full | DKS_F_SILU, "silu")?;
                }
            }
            run(&[md], std::slice::from_ref(&r), &x, Some(&u), m, gs, served_nst, DKS_F_SUH | DKS_F_SILU, "silu partials")?;
            ran.push(m);
        }
        println!("      silu prologue K={k} N={n} b{bits}: bitwise == xq_silu_mul + old chain, M {}{} ({})", m_ranges(&ran),
                 if skipped.is_empty() { String::new() } else { format!(" [M {} skipped: no feasible grid]", m_ranges(&skipped)) },
                 md.name);
    }
    if launches == 0 {
        bail!("EXL3-DENSE FAIL: no DENSE launch ran ({nskip} cell(s) skipped: no feasible grid) — harness failure, nothing was tested");
    }
    println!("    EXL3-DENSE: PASS ({launches} launches bitwise == the old chains: {} single shape(s), {pairs} pair(s), \
              {groups} multi group(s), silu; partials + y + counters; {nskip} (job, M) cell(s) skipped: no feasible grid, \
              the served path falls back to the old chain — {nfb} pair fallback(s) run and bitwise == the old chains)",
             idx_of.len());
    Ok(())
}

