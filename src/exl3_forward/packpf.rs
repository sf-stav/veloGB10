//! PACKPF probe (`--probe-exl3-packpf`): packed multi-prompt prefill vs per-prompt prefill.
//!
//! Idea and first implementation: GitHub andersc (veloGB10 fork, PACK-PF, `--probe-exl3-packpf`).
//! This tree's version takes its parameters as CLI flags and extends the comparison to multi-chunk
//! prompts: pass k of the packed arm carries chunk k of every prompt still prefilling, so segments
//! start at pos0 > 0 and rows past the QSA switch (2,051 on Flash-Next) run the sparse arm.
//!
//! Arms (K prompts, slots 0..K, each prompt chunked at `--packpf-chunk` like the serve grid from 0):
//!   A  per prompt, chunk by chunk (prefill_segs with ONE segment = prefill_chunk, the served path)
//!   B  packed: pass k = chunk k of every prompt that has one (prefill_segs, 2+ segments)
//!   A2 A again (run-to-run determinism of the reference)
//! Compared bit for bit: every GDN conv / scan state, every trunk K/V byte, the draft head's K/V, the
//! PLE conv state, the trunk and head QSA key / pooled planes, every slot's head-fill carry, and the
//! seam step's logits (all K last prompt tokens in one decode step). Timed: A vs B, `--packpf-reps`.
//! RESULT: PACKPF_EXACT | PACKPF_MISMATCH (exit 1).

use super::*;

fn opt_text(name: crate::opts::OptId, def: &str) -> String {
    crate::opts::var(name).ok().filter(|v| !v.is_empty() && v != "unset").unwrap_or_else(|| def.to_string())
}

/// The chunk ends of a fresh prompt prefill over [0, n) at width c (the plain grid, no checkpoint
/// split): every chunk is c rows but the last.
fn grid(n: usize, c: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut p = 0;
    while p < n { let e = (p + c).min(n); v.push((p, e)); p = e; }
    v
}

pub fn probe_pack_pf(a: &FwdArgs) -> Result<bool> {
    let lens: Vec<usize> = opt_text(crate::opt!("packpf-lens"), "1240,1180,1120")
        .split(',').filter_map(|x| x.trim().parse().ok()).filter(|&n: &usize| n >= 2).collect();
    anyhow::ensure!(lens.len() >= 2, "--packpf-lens needs at least two prompt lengths >= 2");
    let chunk: usize = opt_text(crate::opt!("packpf-chunk"), "2048").parse().unwrap_or(2048).max(16);
    let reps: usize = opt_text(crate::opt!("packpf-reps"), "3").parse().unwrap_or(3).max(1);
    let k = lens.len();
    let max_pos = a.max_pos.max(lens.iter().max().copied().unwrap_or(0) + 64);
    let model = FwdModel::load(&a.dir, k, max_pos)?;
    if let Some(why) = model.pack_pf_unsupported() {
        crate::rprintln!("[packpf] this model / option set does not pack: {why}");
        crate::rprintln!("RESULT: PACKPF_UNSUPPORTED");
        return Ok(false);
    }
    // prompt rows = len - 1 (the last token is the seam decode step, as served)
    let grids: Vec<Vec<(usize, usize)>> = lens.iter().map(|&n| grid(n - 1, chunk)).collect();
    let npass = grids.iter().map(|g| g.len()).max().unwrap_or(0);
    let pass_rows: Vec<usize> = (0..npass).map(|p| grids.iter().filter_map(|g| g.get(p)).map(|(s, e)| e - s).sum()).collect();
    let rows = pass_rows.iter().copied().max().unwrap_or(chunk).max(chunk);
    let mut psc = model.prefill_scratch(rows)?;
    let vocab = model.cfg.vocab_size as u64;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let prompts: Vec<Vec<i32>> = lens.iter().map(|&n| (0..n).map(|_| {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        (1000 + seed % (vocab.min(150_000) - 1000)) as i32
    }).collect()).collect();
    crate::rprintln!("[packpf] K={k} prompts, prompt lengths {lens:?}, chunk {chunk}: {npass} packed pass(es), rows per pass {pass_rows:?} (scratch {rows} rows), qsa switch {}",
                     model.qsa_limit());
    let reset = |m: &FwdModel| -> Result<()> { for j in 0..k { m.reset_slot_for_prefill(j)?; } m.dev.synchronize()?; Ok(()) };
    let rw = model.cfg.hidden_size * model.cfg.hc_count.max(1);
    let run_a = |psc: &mut PrefillScratch| -> Result<f64> {
        reset(&model)?;
        let mut busy = 0f64;
        for (j, p) in prompts.iter().enumerate() {
            for &(s, e) in &grids[j] {
                let t = std::time::Instant::now();
                model.prefill_segs(psc, &p[s..e], &[PfSeg { r0: 0, n: e - s, pos0: s, slot: j }])?;
                model.dev.synchronize()?;
                busy += t.elapsed().as_secs_f64();
            }
        }
        Ok(busy * 1e3)
    };
    let run_b = |psc: &mut PrefillScratch| -> Result<f64> {
        reset(&model)?;
        let mut busy = 0f64;
        for p in 0..npass {
            let mut toks = Vec::new();
            let mut segs = Vec::new();
            for (j, g) in grids.iter().enumerate() {
                if let Some(&(s, e)) = g.get(p) {
                    segs.push(PfSeg { r0: toks.len(), n: e - s, pos0: s, slot: j });
                    toks.extend_from_slice(&prompts[j][s..e]);
                }
            }
            let t = std::time::Instant::now();
            model.prefill_segs(psc, &toks, &segs)?;
            model.dev.synchronize()?;
            busy += t.elapsed().as_secs_f64();
        }
        Ok(busy * 1e3)
    };
    type Snap = Vec<(String, Vec<u8>)>;
    let bytes_f32 = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect() };
    let bytes_u16 = |v: Vec<u16>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let snap = |psc: &PrefillScratch| -> Result<Snap> {
        model.dev.synchronize()?;
        let mut s: Snap = Vec::new();
        for (i, x) in model.conv_state.iter().enumerate() { s.push((format!("conv{i}"), bytes_f32(model.dev.dtoh_sync_copy(x)?))); }
        for (i, x) in model.s_state.iter().enumerate() { s.push((format!("gdnS{i}"), bytes_f32(model.dev.dtoh_sync_copy(x)?))); }
        for (i, x) in model.k_cache.iter().enumerate() { s.push((format!("kv{i}"), model.dev.dtoh_sync_copy(x)?)); }
        s.push(("ple_state".into(), bytes_u16(model.dev.dtoh_sync_copy(&model.ple_state)?)));
        for (i, x) in model.qsa_keys.iter().enumerate() { s.push((format!("qsa_keys{i}"), bytes_u16(model.dev.dtoh_sync_copy(x)?))); }
        for (i, x) in model.qsa_pool.iter().enumerate() { s.push((format!("qsa_pool{i}"), bytes_u16(model.dev.dtoh_sync_copy(x)?))); }
        if let Some(hd) = model.mtp.as_ref() {
            s.push(("head_kv".into(), model.dev.dtoh_sync_copy(&hd.kv)?));
            if let Some(x) = hd.qsa_keys.as_ref() { s.push(("head_qsa_keys".into(), bytes_u16(model.dev.dtoh_sync_copy(x)?))); }
            if let Some(x) = hd.qsa_pool.as_ref() { s.push(("head_qsa_pool".into(), bytes_u16(model.dev.dtoh_sync_copy(x)?))); }
            let c = model.dev.dtoh_sync_copy(&psc.hfill_carry_slots)?;
            s.push(("carries".into(), bytes_f32(c[..k * rw].to_vec())));
        }
        Ok(s)
    };
    // the seam step: every slot's last prompt token in ONE forward_step (m = K) -> logits rows
    let mut dsc = Scratch::new(&model.dev, &model.cfg, k, model.cfg.rotary_dim, model.max_pos)?;
    let vsz = model.cfg.vocab_size;
    let mut seam = |dsc: &mut Scratch| -> Result<(Vec<u16>, Vec<i32>)> {
        let toks: Vec<i32> = prompts.iter().map(|p| p[p.len() - 1]).collect();
        let pos: Vec<usize> = prompts.iter().map(|p| p.len() - 1).collect();
        let slots: Vec<usize> = (0..k).collect();
        let ids = model.forward_step(dsc, &toks, &pos, &slots, None)?;
        model.dev.synchronize()?;
        let lg = model.dev.dtoh_sync_copy(&dsc.logits)?;
        Ok((lg[..k * vsz].to_vec(), ids.iter().map(|&x| x as i32).collect()))
    };
    // warm-up both arms (cold PLE pages, cuBLASLt plans, autotune)
    run_a(&mut psc)?;
    run_b(&mut psc)?;
    // seam first, then the snapshot: the seam step writes K/V row len-1 of every slot, which no
    // prefill rewrites — snapshotting before it would compare the previous arm's seam rows
    run_a(&mut psc)?; let la = seam(&mut dsc)?; let sa = snap(&psc)?;
    run_b(&mut psc)?; let lb = seam(&mut dsc)?; let sb = snap(&psc)?;
    run_a(&mut psc)?; let la2 = seam(&mut dsc)?; let sa2 = snap(&psc)?;
    let diff = |x: &Snap, y: &Snap| -> Vec<(String, usize, usize)> {
        x.iter().zip(y).map(|((n, p), (_, q))| (n.clone(), p.iter().zip(q).filter(|(u, v)| u != v).count() + p.len().abs_diff(q.len()), p.len()))
            .filter(|t| t.1 > 0).collect()
    };
    let d_ab = diff(&sa, &sb);
    let d_aa = diff(&sa, &sa2);
    let total_bytes: usize = sa.iter().map(|(_, v)| v.len()).sum();
    let ldiff = |x: &(Vec<u16>, Vec<i32>), y: &(Vec<u16>, Vec<i32>)| -> (usize, usize) {
        (x.0.iter().zip(&y.0).filter(|(u, v)| u != v).count(), x.1.iter().zip(&y.1).filter(|(u, v)| u != v).count())
    };
    let (lab, iab) = ldiff(&la, &lb);
    let (laa, iaa) = ldiff(&la, &la2);
    crate::rprintln!("[packpf] A vs A2 (repeat): state planes differing {} ({:?}); seam logits differ {laa} of {}, argmax differ {iaa}",
                     d_aa.len(), d_aa.iter().take(6).collect::<Vec<_>>(), k * vsz);
    crate::rprintln!("[packpf] A vs B (packed): state planes differing {} of {} ({} bytes compared){}; seam logits differ {lab} of {}, argmax differ {iab}; first tokens A {:?} B {:?}",
                     d_ab.len(), sa.len(), total_bytes,
                     if d_ab.is_empty() { String::new() } else { format!(" {:?}", d_ab.iter().take(8).collect::<Vec<_>>()) },
                     k * vsz, la.1, lb.1);
    let mut ta = Vec::new();
    let mut tb = Vec::new();
    for _ in 0..reps { ta.push(run_a(&mut psc)?); tb.push(run_b(&mut psc)?); }
    ta.sort_by(|x, y| x.partial_cmp(y).unwrap());
    tb.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (ma, mb) = (ta[ta.len() / 2], tb[tb.len() / 2]);
    let total: usize = lens.iter().map(|n| n - 1).sum();
    crate::rprintln!("[packpf] TIME per-prompt {ma:.1} ms ({:.0} tok/s)  packed {mb:.1} ms ({:.0} tok/s)  speedup {:.3}x  (medians of {reps}; A {ta:.1?} B {tb:.1?})",
                     total as f64 / ma * 1e3, total as f64 / mb * 1e3, ma / mb);
    let exact = d_ab.is_empty() && lab == 0 && iab == 0;
    crate::rprintln!("RESULT: {}", if exact { "PACKPF_EXACT" } else { "PACKPF_MISMATCH" });
    Ok(exact)
}
