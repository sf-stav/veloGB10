//! `probe_prefill_x` (chunked prefill against token-by-token forward ground truth), `probe_pfx_trace`, and the
//! packed-verify probe entry points with their `PackShadow` / `Pack2*` helper types. Probe-only.
//! See `mod.rs` for the file map.

use super::*;

/// `--probe-exl3-prefillx`: chunked-prefill vs token-by-token equivalence on the
/// EXL3 engine (S-A3-f-d serve-text investigation). For several chunk widths C,
/// runs the serve's exact prefill arm (prefill_chunk sweeps + prefill_argmax)
/// and compares the first generated token and the final slot-0 conv/GDN state
/// against the reference token-by-token forward_step sweep. Any mismatch is a
/// serve-correctness bug that prose smokes cannot see (chat derailed through
/// prefill_chunk while the same ids through forward_step were exact).
pub fn probe_prefill_x(a: &FwdArgs) -> Result<()> {
    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    let mut sc = Scratch::new(&model.dev, &model.cfg, 1, model.cfg.rotary_dim, model.max_pos)?;
    let ids = gate_prompt_ids(a)?;
    let n = ids.len();
    crate::dprintln!("EXL3 prefill-equivalence probe: prompt={n} tokens");

    // S-A3-r: with --prefillx-qsa-keys the indexer key planes are zeroed before each arm, so
    // each arm's dump shows only its own writes (the planes are otherwise never reset).
    let zero_qsa_keys = || -> Result<()> {
        if crate::opts::var(crate::opt!("prefillx-qsa-keys")).is_err() { return Ok(()); }
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        for k in model.qsa_keys.iter() {
            let n = k.len() as i64;
            xqlaunch!(l, "xq_memset_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                     (*k.device_ptr() as u64, n, 0i32))?;
        }
        model.dev.synchronize()?;
        Ok(())
    };
    // Reference arm: token-by-token forward_step sweeps (decode-kernel path).
    zero_qsa_keys()?;
    model.reset_slot(0)?;
    for (i, t) in ids.iter().enumerate() {
        model.forward_step(&mut sc, &[*t as i32], &[i], &[0], None)?;
    }
    let arg_ref = model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32;
    let st_ref = snap_slot0_state(&model)?;
    // L10: --prefillx-tier=1 runs the Q1 tolerance tier (implies the relL2 report and the
    // bit-0 ulp floor); --prefillx-expect-sig=<hex> checks the pinned signature.
    let tier_on = crate::opts::var(crate::opt!("prefillx-tier")).map_or(false, |v| v != "0" && !v.is_empty());
    let expect_sig: Option<String> = crate::opts::var(crate::opt!("prefillx-expect-sig")).ok()
        .map(|v| v.trim().to_ascii_lowercase()).filter(|v| !v.is_empty());
    // the signature's lines, in print order (reference / chunk C= / the absolute RESULT line)
    let mut sig_text = String::new();
    let rel_on = crate::opts::var(crate::opt!("prefillx-rel")).is_ok() || tier_on;
    let kv_ref = if rel_on { snap_slot0_kv(&model, n)? } else { Vec::new() };
    // S-A3-q: the reference's top-3 logits — a first-token MISMATCH inside a near-tie margin
    // is the drift class; outside it, a real divergence.
    let lg = model.dev.dtoh_sync_copy(&sc.logits)?;
    let v = model.cfg.vocab_size.min(lg.len());
    let mut top: Vec<(f32, usize)> = lg[..v].iter().enumerate()
        .map(|(i, &b)| (half::f16::from_bits(b).to_f32(), i)).collect();
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let line = format!("reference (token-by-token forward_step): first={arg_ref}  top3 logits {:?}",
                       &top[..3.min(top.len())]);
    crate::dprintln!("{line}");
    sig_text += &line; sig_text.push('\n');
    // S-A3-r diagnostic --prefillx-qsa-keys=1: which positions of QSA layer 0's raw indexer
    // key plane (slot 0) each arm has written (non-zero rows), and key diffs where both wrote.
    let qsa_keys_dump = |tag: &str, prev: Option<&Vec<u16>>| -> Result<Option<Vec<u16>>> {
        if crate::opts::var(crate::opt!("prefillx-qsa-keys")).is_err() || model.qsa_keys.is_empty() { return Ok(None); }
        let hdx = model.cfg.indexer_head_dim;
        let v = model.dev.dtoh_sync_copy(&model.qsa_keys[0])?;
        let rows = (v.len() / hdx).min(n);
        let nz: Vec<usize> = (0..rows).filter(|&r| v[r * hdx..(r + 1) * hdx].iter().any(|&b| b != 0)).collect();
        let (lo, hi) = (nz.first().copied(), nz.last().copied());
        let mut msg = format!("QSA_KEYS[{tag}] layer0 slot0: written rows {} of {rows} (first {:?}, last {:?})", nz.len(), lo, hi);
        if let Some(pv) = prev {
            let (mut both, mut differ, mut only_this, mut only_ref) = (0, 0, 0, 0);
            for r in 0..rows {
                let a = &pv[r * hdx..(r + 1) * hdx]; let b = &v[r * hdx..(r + 1) * hdx];
                let (za, zb) = (a.iter().any(|&x| x != 0), b.iter().any(|&x| x != 0));
                match (za, zb) { (true, true) => { both += 1; if a != b { differ += 1; } }
                                 (false, true) => only_this += 1, (true, false) => only_ref += 1, _ => {} }
            }
            msg += &format!("; vs reference: both {both} (bit-differ {differ}), only-this {only_this}, only-ref {only_ref}");
        }
        crate::dprintln!("{msg}");
        Ok(Some(v))
    };
    let keys_ref = qsa_keys_dump("reference", None)?;
    let dbg_ref: Option<(Vec<i32>, Vec<f32>)> = match model.qsa_dbg.as_ref() {
        Some((_, ds, dc, _, _)) => Some((model.dev.dtoh_sync_copy(ds)?, model.dev.dtoh_sync_copy(dc)?)),
        None => None,
    };
    // S-A3-r: the last QSA layer's selection list + block scores for the final prompt row.
    let qsa_on_dump = crate::opts::var(crate::opt!("prefillx-qsa-keys")).is_ok() && model.cfg.has_indexer();
    let sel_pitch = model.qsa_limit();
    let nbs = (model.max_pos / model.cfg.indexer_compress_ratio).max(1);
    let sel_ref: Option<(Vec<i32>, Vec<f32>)> = if qsa_on_dump {
        let sel = model.dev.dtoh_sync_copy(&sc.qsa_sel)?;
        let ps = model.dev.dtoh_sync_copy(&sc.qsa_psel)?;
        let scv = model.dev.dtoh_sync_copy(&sc.qsa_scores)?;
        let cnt = (ps[0] + 1).max(0) as usize;
        Some((sel[..cnt.min(sel_pitch)].to_vec(), scv[..nbs].to_vec()))
    } else { None };

    // Chunked arm: the serve's exact loop at chunk width C.
    let mut all_ok = true;
    // S-A3-q: --prefillx-widths=a,b,c overrides the width list (serve widths 256/2048).
    // S-A3-s: an entry "A+B" = irregular split (first chunk A, then B-token chunks).
    // TP-4X1: an entry "g<C>" / "g<C>a" runs the SERVED chunk grid of this prompt (the real
    // plan_grid: tail checkpoint at message boundary - 8 = n - 14, merge cap as under TP; trailing
    // 'a' = the removed --prefill-absorb-tail), so the old / steady-4095 / absorbed grids are probed as served.
    let served_grid = |c: usize, absorb: bool| -> Vec<usize> {
        let tail = wp16::tail_point(Some(n.saturating_sub(6)), 0, n, c, tail_ckpt_extra_chunk());
        wp16::plan_grid(0, n, c, true, tail, wp16::merge_cap(c, true), absorb.then(recon_min_rows), true)
    };
    let widths: Vec<(usize, usize, Option<(String, Vec<usize>)>)> = crate::opts::var(crate::opt!("prefillx-widths")).ok()
        .map(|v| v.split(',').filter_map(|x| {
            let x = x.trim();
            if let Some(g) = x.strip_prefix('g') {
                let (num, absorb) = match g.strip_suffix('a') { Some(h) => (h, true), None => (g, false) };
                let c: usize = num.parse().ok()?;
                let ends = served_grid(c.min(n), absorb);
                let mut prev = 0usize;
                let rows: Vec<usize> = ends.iter().map(|&e| { let r = e - prev; prev = e; r }).collect();
                crate::dprintln!("served grid {x} (n={n}): {} chunk(s) rows {:?}", rows.len(), rows);
                return Some((rows[0], rows.iter().copied().max().unwrap_or(1), Some((x.to_string(), ends))));
            }
            match x.split_once('+') {
                Some((a, b)) => Some((a.parse().ok()?, b.parse().ok()?, None)),
                None => x.parse().ok().map(|c: usize| (c, c, None)),
            }
        }).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 60].into_iter().map(|c| (c, c, None)).collect());
    let dec_rows: Option<(Vec<i32>, Vec<i32>)> = match model.qsa_rowdbg.as_ref() {
        Some((_, ds, dp, _, _)) => Some((model.dev.dtoh_sync_copy(ds)?, model.dev.dtoh_sync_copy(dp)?)),
        None => None,
    };
    // S-A3-u: a width longer than the prompt is CLAMPED to it (one chunk), never skipped — the
    // old `continue` let a 16-token prompt at C=2048 compare nothing and still print EXACT.
    let mut arms = 0usize;
    // L10 tier inputs per chunked arm: (tag, S med, KV med, first-token rank, gap)
    let mut tier_arms: Vec<(String, f64, Option<f64>, usize, f32)> = Vec::new();
    for (c_first, c, explicit) in widths {
        let (c_first, c) = (c_first.min(n), c.min(n));
        arms += 1;
        zero_qsa_keys()?;
        model.reset_slot(0)?;
        let mut psc = model.prefill_scratch(c.max(c_first))?;
        let mut pos = 0usize;
        let mut last0 = 0usize;
        let label = explicit.as_ref().map_or(format!("{c_first:>3}+{c}"), |(l, _)| l.clone());
        let mut ex_it = explicit.as_ref().map(|(_, v)| v.iter().copied());
        while pos < n {
            let w = if pos == 0 { c_first } else { c };
            let end = match ex_it.as_mut() { Some(it) => it.next().context("explicit grid ended before n")?, None => (pos + w).min(n) };
            let tk: Vec<i32> = ids[pos..end].iter().map(|&t| t as i32).collect();
            last0 = pos;
            model.prefill_chunk(&mut psc, &tk, pos, 0, end == n, None)?;
            pos = end;
        }
        let arg_c = model.prefill_argmax(&psc)?;
        if let (Some((dsel, dps)), Some((k, _, _, ps, pp))) = (dec_rows.as_ref(), model.qsa_rowdbg.as_ref()) {
            // per-row selection diff over every sparse position both paths recorded
            let psel_ = model.dev.dtoh_sync_copy(ps)?; let pps = model.dev.dtoh_sync_copy(pp)?;
            let sm = model.qsa_limit();
            let (mut nrows, mut ndiff, mut first, mut tot) = (0usize, 0usize, None, 0usize);
            let mut per_chunk: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
            for p in (model.qsa_limit() + 1)..n.min(QSA_ROWDUMP_CAP) {
                if dps[p] < 0 || pps[p] < 0 { continue; }
                nrows += 1;
                let a: std::collections::BTreeSet<i32> = dsel[p * sm..p * sm + (dps[p] + 1) as usize].iter().copied().collect();
                let b: std::collections::BTreeSet<i32> = psel_[p * sm..p * sm + (pps[p] + 1) as usize].iter().copied().collect();
                let d = a.symmetric_difference(&b).count();
                let ch = if p < c_first { 0 } else { 1 + (p - c_first) / c };
                let e = per_chunk.entry(ch).or_insert((0, 0)); e.0 += 1;
                if d > 0 { ndiff += 1; tot += d; e.1 += 1; if first.is_none() { first = Some((p, d)); } }
            }
            crate::dprintln!("QSA_ROWS[C={c_first}+{c}] layer {k}: sparse rows compared {nrows}, rows with different selection {ndiff} (first {:?}, total sym-diff {tot}); per chunk (rows, differing): {:?}",
                     first, per_chunk);
        }
        if let (Some((sr, scr)), Some((k, _, _, ps, pcs))) = (dbg_ref.as_ref(), model.qsa_dbg.as_ref()) {
            let r = n - 1 - last0;
            let sp = model.qsa_limit();
            let sel = model.dev.dtoh_sync_copy(ps)?; let scv = model.dev.dtoh_sync_copy(pcs)?;
            let row = &sel[r * sp..(r + 1) * sp]; let srow = &scv[r * nbs..(r + 1) * nbs];
            let a: std::collections::BTreeSet<i32> = sr.iter().copied().filter(|&x| x >= 0).collect();
            let b: std::collections::BTreeSet<i32> = row.iter().copied().filter(|&x| x >= 0).collect();
            let nb = n / model.cfg.indexer_compress_ratio;
            let mut sorted: Vec<f32> = scr[..nb].to_vec(); sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
            let kth = model.cfg.indexer_budget / model.cfg.indexer_compress_ratio;
            let (mut md, mut mean) = (0f32, 0f64);
            for i in 0..nb { let d = (scr[i] - srow[i]).abs(); md = md.max(d); mean += d as f64; }
            crate::dprintln!("QSA_DUMP[C={c}] QSA layer {k} final row: sel sym-diff {} positions; score |diff| max {:.4} mean {:.4}; ref score @rank {kth} {:.4} vs @rank {} {:.4} (selection margin {:.4})",
                     a.symmetric_difference(&b).count(), md, mean / nb as f64,
                     sorted[kth - 1], kth + 1, sorted[kth], sorted[kth - 1] - sorted[kth]);
        }
        if let Some((sr, scr)) = sel_ref.as_ref() {
            let r = n - 1 - last0;
            let sel = model.dev.dtoh_sync_copy(&psc.qsa_sel)?;
            let ps = model.dev.dtoh_sync_copy(&psc.qsa_psel)?;
            let scv = model.dev.dtoh_sync_copy(&psc.qsa_scores)?;
            let cnt = (ps[r] + 1).max(0) as usize;
            let sc_ = &sel[r * sel_pitch..r * sel_pitch + cnt.min(sel_pitch)];
            let scc = &scv[r * nbs..(r + 1) * nbs];
            let a: std::collections::BTreeSet<i32> = sr.iter().copied().collect();
            let b: std::collections::BTreeSet<i32> = sc_.iter().copied().collect();
            let only_ref: Vec<i32> = a.difference(&b).copied().take(12).collect();
            let only_chk: Vec<i32> = b.difference(&a).copied().take(12).collect();
            let nb = ((n) / model.cfg.indexer_compress_ratio).min(nbs);
            let (mut maxd, mut arg) = (0f32, 0usize);
            for i in 0..nb { let d = (scr[i] - scc[i]).abs(); if d.is_finite() && d > maxd { maxd = d; arg = i; } }
            // S-A3-r scorer twin: SAME key plane (last QSA layer), SAME normed/roped q row (the
            // prefill's final row), SAME position -> decode qsa_score_b vs the prefill scores.
            {
                let cfg = &model.cfg;
                let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
                let qk_dim = (heads + 1) * hdx;
                let last_att = model.qsa_keys.len() - 1;
                let params_p = model.layers.iter().filter_map(|ly| match &ly.mixer {
                    Mixer::Attn(a) => a.idx.as_ref().and_then(|i| i.params.as_ref()).map(|p| *p.device_ptr() as u64),
                    _ => None }).last().unwrap();
                let l = Launcher { dev: &model.dev, stream: &model.stream };
                let qrow = model.dev.dtoh_sync_copy(&psc.qsa_qkb)?[r * qk_dim..(r + 1) * qk_dim].to_vec();
                let mut qv = vec![0u16; sc.qsa_qkb.len()]; qv[..qk_dim].copy_from_slice(&qrow);
                model.dev.htod_copy_into(qv, &mut sc.qsa_qkb)?;
                let mut pv = vec![0i32; sc.pos.len()]; pv[0] = (n - 1) as i32;
                model.dev.htod_copy_into(pv, &mut sc.pos)?;
                model.dev.htod_copy_into(vec![0i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
                let keys_p = *model.qsa_keys[last_att].device_ptr() as u64;
                let geom: u64 = crate::gpu::pack_qsa_geom(model.max_pos, qk_dim, nbs);
                gblaunch!(l, "qsa_score_b", ((nbs.div_ceil(8)) as u32, 1u32, 1), (256, 1, 1), 0,
                          (&mut sc.qsa_scores, &sc.qsa_qkb, keys_p, &sc.pos, &sc.slot_ids,
                           0u64, 0u64, params_p, geom as i64))?;
                model.dev.synchronize()?;
                let dsc = model.dev.dtoh_sync_copy(&sc.qsa_scores)?;
                let (mut md, mut ai, mut cnt_big) = (0f32, 0usize, 0usize);
                for i in 0..nb {
                    let d = (dsc[i] - scc[i]).abs();
                    if d.is_finite() { if d > md { md = d; ai = i; } if d > 1e-2 { cnt_big += 1; } }
                }
                crate::dprintln!("QSA_SCORER_TWIN[C={c}] identical inputs (pos {}, {} blocks): decode-vs-prefill max|diff| {:.5} @blk {} (decode {:.4} prefill {:.4}); blocks with |diff|>1e-2: {}",
                         n - 1, nb, md, ai, dsc[ai], scc[ai], cnt_big);
            }
            crate::dprintln!("QSA_SEL[C={c}] final row, last QSA layer: ref {} entries, chunk {} entries; sym-diff {} (only-ref {:?}, only-chunk {:?}); block-score max|diff| {:.4} @blk {} (ref {:.4} chunk {:.4}); ref first/last {:?}/{:?} chunk first/last {:?}/{:?}",
                     sr.len(), sc_.len(), a.symmetric_difference(&b).count(), only_ref, only_chk, maxd, arg,
                     scr[arg], scc[arg], sr.first(), sr.last(), sc_.first(), sc_.last());
        }
        let _ = qsa_keys_dump(&format!("chunk C={c_first}+{c}"), keys_ref.as_ref())?;
        let st_c = snap_slot0_state(&model)?;
        let (maxd, neq) = state_diff(&st_ref, &st_c);
        let mut rel_meds = (f64::NAN, None);
        if rel_on {
            let kv_c = snap_slot0_kv(&model, n)?;
            rel_meds = prefillx_rel_report(&format!("C={}", label.trim()), &st_ref, &st_c, &kv_ref, &kv_c);
        }
        let am = arg_c != arg_ref;
        let ok = !am && neq == 0;
        all_ok &= ok;
        // S-A3-s: where the chunked first token sits in the REFERENCE's logits (rank, gap to top-1).
        let rk = top.iter().position(|&(_, i)| i as u32 == arg_c);
        if let Some(rk) = rk {
            crate::dprintln!("  chunked first={arg_c}: reference rank {} logit {:.4} (gap to ref top-1 {:.4})",
                     rk + 1, top[rk].0, top[0].0 - top[rk].0);
        }
        let (rank, gap) = rk.map_or((usize::MAX, f32::INFINITY), |r| (r + 1, top[0].0 - top[r].0));
        tier_arms.push((format!("C={}", label.trim()), rel_meds.0, rel_meds.1, rank, gap));
        let line = format!("chunk C={label}: first={arg_c} ({}), state maxd={maxd:.7} bit-neq={neq}  {}",
                           if am { "MISMATCH" } else { "match" }, if ok { "OK" } else { "DIVERGES" });
        crate::dprintln!("{line}");
        sig_text += &line; sig_text.push('\n');
    }
    // Q1 noise floor (--prefillx-ulp=none,0,13,...): re-run the token-by-token REFERENCE,
    // injecting after every token but the last a pseudo-random 1-bit flip at mantissa bit k into
    // ~half the elements of every GDN recurrent state (k=0: one f32 ulp; k=13: ~one f16 ulp,
    // relative 2^-10). "none" = the plain rerun (determinism control). The drift this produces
    // is what pure rounding-level perturbation of the same recurrence amplifies to.
    let mut ulp_list: Option<String> = crate::opts::var(crate::opt!("prefillx-ulp")).ok();
    if tier_on && !ulp_list.as_deref().map_or(false, |l| l.split(',').any(|x| x.trim() == "0")) {
        ulp_list = Some(match ulp_list { Some(l) if !l.trim().is_empty() => format!("{l},0"), _ => "0".into() });
    }
    let mut floor0: Option<(f64, Option<f64>)> = None;
    if let Some(list) = ulp_list {
        let cfg = &model.cfg;
        let per_s = cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        for ent in list.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
            let bit: Option<u32> = if ent == "none" { None } else { ent.parse().ok() };
            zero_qsa_keys()?;
            model.reset_slot(0)?;
            for (i, t) in ids.iter().enumerate() {
                model.forward_step(&mut sc, &[*t as i32], &[i], &[0], None)?;
                let Some(k) = bit else { continue };
                if i + 1 == n || k > 22 { continue; }
                for (li, ss) in model.s_state.iter().enumerate() {
                    let mut h: Vec<f32> = model.dev.dtoh_sync_copy(ss)?;
                    for (j, x) in h[..per_s].iter_mut().enumerate() {
                        let mut z = (j as u64) ^ ((i as u64) << 40) ^ ((li as u64) << 52);
                        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                        z ^= z >> 31;
                        let b = x.to_bits();
                        if z & 1 == 1 && (b & 0x7F80_0000) != 0 { *x = f32::from_bits(b ^ (1u32 << k)); }
                    }
                    let tmp = model.dev.htod_copy(h[..per_s].to_vec())?;
                    let dst = *ss.device_ptr() as u64;
                    xqlaunch!(l, "xq_copy_f32", (((per_s + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                              (dst, &tmp, per_s as i64, 0i64, 0i64))?;
                    model.dev.synchronize()?;
                }
            }
            let arg_u = model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32;
            let st_u = snap_slot0_state(&model)?;
            let (maxd, neq) = state_diff(&st_ref, &st_u);
            if rel_on {
                let kv_u = snap_slot0_kv(&model, n)?;
                let m = prefillx_rel_report(&format!("ULP bit={ent}"), &st_ref, &st_u, &kv_ref, &kv_u);
                if bit == Some(0) { floor0 = Some(m); }
            }
            crate::dprintln!("ULP-FLOOR bit={ent}: first={arg_u} ({}), state maxd={maxd:.7} bit-neq={neq}",
                     if arg_u == arg_ref { "match" } else { "MISMATCH" });
        }
    }
    anyhow::ensure!(arms > 0, "probe_prefill_x compared no chunked arm (empty width list)");
    let line = format!("RESULT: {} (exl3 probe_prefill_x, {arms} arm(s) compared)", if all_ok { "EXACT" } else { "DIVERGES" });
    crate::dprintln!("{line}");
    sig_text += &line; sig_text.push('\n');
    // L10 (a) the pinned signature: md5 of the reference / chunk C= / absolute RESULT lines, first
    // 12 hex = exactly the historical `grep -E "^chunk C=|^RESULT|^reference" | md5sum | cut -c1-12`
    // (this line and the verdicts below match none of those patterns except the new RESULT lines,
    // which come AFTER — scripts that grep the log see extra ^RESULT lines only when they opt in).
    let sig = md5_hex(sig_text.as_bytes())[..12].to_string();
    crate::dprintln!("PREFILLX_SIG {sig} (md5[:12] of the reference/chunk/RESULT lines; {arms} arm(s))");
    let mut new_ok = true;
    if let Some(want) = expect_sig.as_ref() {
        let ok = *want == sig;
        new_ok &= ok;
        crate::dprintln!("RESULT: {} (signature {sig}, expected {want})", if ok { "PREFILLX_SIG_OK" } else { "PREFILLX_SIG_MISMATCH" });
    }
    // L10 (b) the Q1 tolerance tier
    if tier_on {
        let mut ok = true;
        match floor0 {
            None => { ok = false; crate::dprintln!("PREFILLX_TIER no bit-0 floor measured (relL2 report unavailable)"); }
            Some((fs, fkv)) => {
                let fkv_s = fkv.map_or("n/a".to_string(), |v| format!("{v:.3e}"));
                crate::dprintln!("PREFILLX_TIER floor (bit 0): S relL2 med {fs:.3e} | KV relL2 med {fkv_s}");
                for (tag, s_m, kv_m, rank, gap) in &tier_arms {
                    let fails = prefillx_tier_fails(*s_m, *kv_m, fs, fkv, *rank, *gap);
                    let kv_x = match (kv_m, fkv) { (Some(a), Some(b)) => format!("{:.2}x", a / b), _ => "n/a".into() };
                    println!("PREFILLX_TIER[{tag}] S {:.2}x KV {kv_x} of floor | first rank {rank} gap {gap:.4} -> {}",
                             s_m / fs, if fails.is_empty() { "ok".to_string() } else { format!("FAIL ({})", fails.join("; ")) });
                    ok &= fails.is_empty();
                }
            }
        }
        new_ok &= ok;
        println!("RESULT: {} (tolerance tier: S/KV relL2 med <= 3x the bit-0 ulp floor, first token rank <= 5 within 1.0 logit; {arms} arm(s))",
                 if ok { "PREFILLX_TIER_OK" } else { "PREFILLX_TIER_FAIL" });
    }
    // exit code: the legacy absolute verdict, unless the caller asked for the re-baselined checks
    // (the served posture can never be EXACT — Q1) — then they decide.
    let pass = if expect_sig.is_some() || tier_on { new_ok } else { all_ok };
    if !pass { std::process::exit(1); }
    Ok(())
}

/// `--probe-exl3-pfxtrace`: S-A3-f-d night instrument. Diffs the serve's
/// chunked prefill arm (prefill_chunk at C=1, the degenerate width) against
/// the token-by-token reference arm (forward_step) STAGE BY STAGE at token 0
/// (per-layer: embed / hc / qkv-z / a-b / conv / gdn / gate / attn / moe via
/// the LadderCtx dump files), then sweeps the remaining tokens (state
/// snapshots) for the FIRST diverging token position. Classifies the
/// divergence: rounding-class (tiny at the first kernel-family difference,
/// smooth growth) vs semantic (O(1) jump at a named stage = the bug).
pub fn probe_pfx_trace(a: &FwdArgs) -> Result<()> {
    let dir_r = "/tmp/pfx_ref".to_string();
    let dir_c = "/tmp/pfx_chk".to_string();
    for d in [&dir_r, &dir_c] {
        let _ = std::fs::remove_dir_all(d);
        std::fs::create_dir_all(d)?;
    }
    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    let mut sc = Scratch::new(&model.dev, &model.cfg, 1, model.cfg.rotary_dim, model.max_pos)?;
    let ids = gate_prompt_ids(a)?;
    let nl = model.layers.len();
    crate::dprintln!("EXL3 prefill TRACE: {} tokens, {nl} layers; stage diff @ token 0 + first-divergent-token sweep", ids.len());

    // ---- Arm R: token-by-token reference (decode kernels), dumps at token 0.
    model.reset_slot(0)?;
    let mut snaps_r: Vec<Vec<u32>> = Vec::new();
    {
        let mut ld = LadderCtx { dir: dir_r.clone(), step: 0, layers: (0..nl).collect(), only_step: Some(0), mem: None, digest: false, sp: None };
        model.forward_step(&mut sc, &[ids[0] as i32], &[0], &[0], Some(&mut ld))?;
    }
    snaps_r.push(flat_state(&snap_slot0_state(&model)?));
    for (i, t) in ids.iter().enumerate().skip(1) {
        model.forward_step(&mut sc, &[*t as i32], &[i], &[0], None)?;
        if snaps_r.len() < 8 { snaps_r.push(flat_state(&snap_slot0_state(&model)?)); }
    }
    let arg_ref = model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32;

    // ---- Arm C: the serve loop at C=1, dumps at token 0.
    model.reset_slot(0)?;
    let mut psc = model.prefill_scratch(1)?;
    let mut snaps_c: Vec<Vec<u32>> = Vec::new();
    {
        let ld = LadderCtx { dir: dir_c.clone(), step: 0, layers: (0..nl).collect(), only_step: Some(0), mem: None, digest: false, sp: None };
        model.prefill_chunk(&mut psc, &[ids[0] as i32], 0, 0, true, Some(&ld))?;
    }
    snaps_c.push(flat_state(&snap_slot0_state(&model)?));
    for (i, t) in ids.iter().enumerate().skip(1) {
        model.prefill_chunk(&mut psc, &[*t as i32], i, 0, i + 1 == ids.len(), None)?;
        if snaps_c.len() < 8 { snaps_c.push(flat_state(&snap_slot0_state(&model)?)); }
    }
    let arg_chk = model.prefill_argmax(&psc)?;

    // ---- Per-token first divergence (state bits, first 8 tokens).
    let mut first_tok: Option<usize> = None;
    for i in 0..snaps_r.len().min(snaps_c.len()) {
        let neq = snaps_r[i].iter().zip(snaps_c[i].iter()).filter(|(a, b)| a != b).count();
        if neq > 0 { first_tok = Some(i); crate::dprintln!("state bits diverge at token {i} (of compared {}): neq={neq}", snaps_r[i].len()); break; }
    }
    if first_tok.is_none() { crate::dprintln!("state bits EXACT for first {} token(s)", snaps_r.len().min(snaps_c.len())); }

    // ---- Stage diff at token 0, execution order.
    let mut names: Vec<String> = std::fs::read_dir(&dir_r)?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|f| f.ends_with("_s000.f32"))
        .map(|f| f.trim_end_matches("_s000.f32").to_string())
        .collect();
    names.sort_by_key(|n| stage_key(n));
    crate::dprintln!("{:<14} {:>4} {:>12} {:>10} {:>10}", "stage", "n", "max|d|", "@idx", "rel");
    let (mut first_any, mut first_sem): (Option<String>, Option<String>) = (None, None);
    for name in &names {
        let cf = format!("{dir_c}/{name}_s000.f32");
        if !std::path::Path::new(&cf).exists() {
            crate::dprintln!("{name:<14} present in reference only");
            continue;
        }
        let rv = read_f32_le(&format!("{dir_r}/{name}_s000.f32"))?;
        let cv = read_f32_le(&cf)?;
        let n = rv.len().min(cv.len());
        if rv.len() != cv.len() { crate::dprintln!("{name:<14} LENGTH MISMATCH {} vs {}", rv.len(), cv.len()); }
        let (mut md, mut mi, mut mg) = (0.0f64, 0usize, 0.0f64);
        for i in 0..n {
            let d = (rv[i] as f64 - cv[i] as f64).abs();
            if d > md { md = d; mi = i; }
            mg = mg.max((rv[i] as f64).abs());
        }
        let rel = if mg > 1e-9 { md / mg } else { md };
        if md == 0.0 { continue; }
        let tag = if rel > 1e-2 && md > 1e-3 { "  <== SEMANTIC" } else { "" };
        crate::dprintln!("{name:<14} {n:>9} {md:>12.4e} {mi:>10} {rel:>10.2e}{tag}");
        if first_any.is_none() { first_any = Some(name.clone()); }
        if first_sem.is_none() && rel > 1e-2 && md > 1e-3 { first_sem = Some(name.clone()); }
    }
    match (&first_any, &first_sem) {
        (a, s) => {
            crate::dprintln!("FIRST DIFF: {}", a.clone().unwrap_or_else(|| "(none — bitwise identical)".into()));
            crate::dprintln!("FIRST SEMANTIC: {}", s.clone().unwrap_or_else(|| "(none above threshold)".into()));
        }
    }
    crate::dprintln!("first token: ref={arg_ref} chk={arg_chk} ({})",
             if arg_ref == arg_chk { "match" } else { "MISMATCH" });
    Ok(())
}

fn flat_state(s: &(Vec<Vec<u32>>, Vec<Vec<u32>>)) -> Vec<u32> {
    let mut v = Vec::new();
    v.extend(s.0.iter().flatten());
    v.extend(s.1.iter().flatten());
    v
}

fn read_f32_le(p: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(p)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Execution-order key for trace stage names (layer, stage-within-layer);
/// global stages sort last.
fn stage_key(name: &str) -> (usize, usize) {
    match name {
        "embed" => return (usize::MAX, 0),
        "precollapse" => return (usize::MAX, 1),
        "finalmix" => return (usize::MAX, 2),
        "logits" => return (usize::MAX, 3),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix('g') {
        if let Some(p) = rest.find('_') {
            let li: usize = rest[..p].parse().unwrap_or(0);
            let pri = match &rest[p + 1..] { "qkv" => 1, "z" => 2, "b" => 3, "a" => 4, "conv" => 5, "core" => 6, "gate" => 7, _ => 8 };
            return (li, pri);
        }
    }
    let dstart = name.rfind(|c: char| !c.is_ascii_digit()).map(|p| p + 1).unwrap_or(0);
    let li: usize = name[dstart..].parse().unwrap_or(0);
    let pri = match &name[..dstart] {
        "xpre" => 0, "rmid" => 10, "yatt" => 11, "xpre_mlp" => 12, "inj" => 13,
        "ysh" => 14, "yd" => 15, "ymlp" => 16, "layer" => 20, _ => 30,
    };
    (li, pri)
}

// ==================== PACK1 LEG 1 (S-B9-CF-PACK1) ====================
/// Max segments in the packed verify (v1: k <= 3 per the design; LEG 1 uses 2).
pub(crate) const PACK_SEG_MAX: usize = 3;

#[derive(Default, Clone)]
/// PACK2: device shadow of every persistent state region the two arms
/// read/write, taken once after the prime (d2d copies — bitwise). Arms restart
/// from this shadow, NEVER from a re-run prime: the chunked prefill's GEMM
/// class runs split-K atomics (batch-invariance binds only the decode/verify
/// classes), so a second prime is NOT bit-equal to the first.
pub(crate) struct PackShadow {
    // (dst device ptr, dst/src word offset of the region, saved words) — the
    // offset is BOTH the save's src offset and the restore's dst offset.
    pub(super) parts: Vec<(u64, i64, CudaSlice<u16>)>,
    pub(super) hist: Vec<i32>,
}

pub(crate) struct Pack2Seg {
    pub core: Vec<u16>,
    pub x: Vec<u16>,
    pub logits: Vec<u16>,
    pub argmax: Vec<i32>,
    pub out: Vec<i32>,
    pub acc: Vec<i32>,
    pub a: usize,
    pub ms: f64,
}

#[derive(Default, Clone)]
pub(crate) struct Pack2Packed {
    pub core_a: Vec<u16>,
    pub core_b: Vec<u16>,
    pub x_a: Vec<u16>,
    pub x_b: Vec<u16>,
    pub logits_a: Vec<u16>,
    pub logits_b: Vec<u16>,
    pub argmax_a: Vec<i32>,
    pub argmax_b: Vec<i32>,
    pub out_a: Vec<i32>,
    pub out_b: Vec<i32>,
    pub acc_a: (i32, i32),
    pub acc_b: (i32, i32),
    pub a_a: usize,
    pub a_b: usize,
    pub ms: f64,
}

pub(crate) fn fnv64_u16(v: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in v {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= (x >> 8) as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub(crate) fn fnv64_i32(v: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in v {
        h ^= (x as u32) as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= ((x as u32) >> 16) as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// PACK1 LEG 1 entry: `--probe-exl3-pack2 --model-dir <pack>`.
/// EAGER 2-request packed verify vs two serial 5-row verifies: bit identity per
/// segment, G3 nostore state diff, a/b/c timing and the ratio r. No graphs, no
/// scheduler, no flag; the served path is untouched (new kernel entries only).
pub fn pack2_probe_exl3(dir: &str, max_pos: usize) -> Result<()> {
    let model = FwdModel::load(dir, 2, max_pos)?; // 2 slots: the packed probe primes 0 and 1
    crate::dprintln!("PACK2: model {} loaded (vocab {}, max_pos {max_pos})", dir, model.cfg.vocab_size);
    model.probe_pack2()
}

/// PACK2 LEG 2 entry: `--probe-exl3-packstate --model-dir <pack>`.
/// The packed ROUND (real draft chains -> one packed verify -> per-segment
/// accept/commit -> per-request re-prime) proven bit-identical to the same
/// rounds run serially (lone verify per request through the served verify_block).
pub fn packstate_probe_exl3(dir: &str, max_pos: usize) -> Result<()> {
    let model = FwdModel::load(dir, 2, max_pos)?; // 2 slots: the probe primes 0 and 1
    crate::dprintln!("PACKSTATE: model {} loaded (vocab {}, max_pos {max_pos})", dir, model.cfg.vocab_size);
    model.probe_packstate()
}


/// PACK3 LEG 3 STEP A entry: `--probe-exl3-pack3 --model-dir <pack>`.
/// Graphed round-pair A/B (V5 lone graph vs P10 packed graph + eager tail),
/// graph-vs-eager bit identity, the 6+4 same-graph replay, the per-stage
/// table and round_pair_ratio.
pub fn pack3_probe_exl3(dir: &str, max_pos: usize) -> Result<()> {
    let model = FwdModel::load(dir, 2, max_pos)?; // 2 slots: slots 0 and 1 pack
    crate::dprintln!("PACK3: model {} loaded (vocab {}, max_pos {max_pos})", dir, model.cfg.vocab_size);
    model.probe_pack3()
}
