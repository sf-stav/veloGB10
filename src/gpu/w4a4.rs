//! `impl GpuModel`: NVFP4 W4A4 prefill (`init_w4a4`, `w4a4_*`, driver `src/w4a4.rs`) and the `--calib-igs` activation tap.
//! Map: `mod.rs`.

use super::*;

// ---------------------------------------------------------------- NVFP4 W4A4 prefill (src/w4a4.rs)
impl GpuModel {
    /// --w4a4-prefill: collect the NVFP4 weights of the requested groups (by qweight pointer),
    /// their `input_global_scale` by name (a fused tensor takes the MIN of its parts = the largest
    /// calibrated amax), size the packed-activation scratch, load the sm_121a module.
    pub(super) fn init_w4a4(&mut self, model_dir: &str) -> anyhow::Result<()> {
        let configured_a4 = std::fs::read_to_string(std::path::Path::new(model_dir).join("config.json"))
            .ok().and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|j| j["quantization_config"]["activation_variant"].as_str().map(str::to_owned))
            .as_deref() == Some("w4a4");
        // An explicit prefill value always wins, including `0`. Narrow decode+verify is a
        // separate experimental opt-in because it changes the target's numerical chain. It is
        // applied to both N=1 and N<=MAX_VERIFY for controlled A/B, but is not lossless end-to-end.
        let prefill_groups = if crate::opts::var_os(crate::opt!("w4a4-prefill")).is_some() {
            crate::w4a4::groups_from_opts().unwrap_or_default()
        } else if configured_a4 {
            vec!["attn".into(), "mlp".into(), "gdn".into(), "lmhead".into()]
        } else { Vec::new() };
        let verify_groups = crate::w4a4::verify_groups_from_opts().unwrap_or_default();
        if prefill_groups.is_empty() && verify_groups.is_empty() { return Ok(()); }
        let mut groups = prefill_groups.clone();
        for g in &verify_groups { if !groups.contains(g) { groups.push(g.clone()); } }
        let narrow_requested = match crate::opts::var(crate::opt!("w4a4-lmhead-narrow")) {
            Ok(v) => v == "1",
            Err(_) => configured_a4,
        };
        anyhow::ensure!(self.mxfp4.is_none(), "--w4a4-prefill/VERIFY and --mxfp4 are exclusive");
        let wide = |g: &str| crate::w4a4::group_on(&prefill_groups, g);
        let narrow = |g: &str| crate::w4a4::group_on(&verify_groups, g);
        let gdn_wide = |part: &str| crate::w4a4::gdn_part_on(&prefill_groups, part);
        let gdn_narrow = |part: &str| crate::w4a4::gdn_part_on(&verify_groups, part);
        let lm = "model.language_model";
        let mut enabled: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut narrow_enabled: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut x_gs: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
        let mut k_max = 0usize;
        {
            let igs = &self.igs_by_name;
            let mut add = |w: &W, names: &[String], wide_on: bool, narrow_on: bool| {
                if !wide_on && !narrow_on { return; }
                if let W::Nvfp4 { qweight, k, m, .. } = w {
                    if *k % 64 != 0 || *m % 16 != 0 { return; }
                    let ptr = *qweight.device_ptr() as u64;
                    if wide_on { enabled.insert(ptr); }
                    if narrow_on { narrow_enabled.insert(ptr); }
                    k_max = k_max.max(*k);
                    let gsx = names.iter().filter_map(|n| igs.get(n)).cloned().fold(f32::INFINITY, f32::min);
                    if gsx.is_finite() && gsx > 0.0 { x_gs.insert(ptr, gsx); }
                }
            };
            for (li, l) in self.layers.iter().enumerate() {
                let lp = format!("{lm}.layers.{li}");
                let (attn_w, attn_n) = (wide("attn"), narrow("attn"));
                if attn_w || attn_n { if let Some(fa) = &l.fa {
                    let (q, k, v) = (format!("{lp}.self_attn.q_proj"), format!("{lp}.self_attn.k_proj"), format!("{lp}.self_attn.v_proj"));
                    match &fa.qkv { AttnIn::Fused(w) => add(w, &[q, k, v], attn_w, attn_n),
                                    AttnIn::Split { q: wq, k: wk, v: wv } => { add(wq, &[q], attn_w, attn_n); add(wk, &[k], attn_w, attn_n); add(wv, &[v], attn_w, attn_n); } }
                    add(&fa.o_proj, &[format!("{lp}.self_attn.o_proj")], attn_w, attn_n);
                    if let Some(ix) = &fa.indexer { add(&ix.qk_proj, &[format!("{lp}.self_attn.indexer.index_qk_proj")], attn_w, attn_n); }
                } }
                let (gdn_in_w, gdn_in_n) = (gdn_wide("gdn-in"), gdn_narrow("gdn-in"));
                let (gdn_out_w, gdn_out_n) = (gdn_wide("gdn-out"), gdn_narrow("gdn-out"));
                if gdn_in_w || gdn_in_n || gdn_out_w || gdn_out_n { if let Some(la) = &l.la {
                    let n = |s: &str| format!("{lp}.linear_attn.{s}");
                    match &la.in_proj {
                        GdnIn::Fused(w) => add(w, &[n("in_proj_qkv"), n("in_proj_z"), n("in_proj_b"), n("in_proj_a")], gdn_in_w, gdn_in_n),
                        GdnIn::Split { qkv, z, b, a } => { add(qkv, &[n("in_proj_qkv")], gdn_in_w, gdn_in_n); add(z, &[n("in_proj_z")], gdn_in_w, gdn_in_n); add(b, &[n("in_proj_b")], gdn_in_w, gdn_in_n); add(a, &[n("in_proj_a")], gdn_in_w, gdn_in_n); }
                    }
                    add(&la.out_proj, &[n("out_proj")], gdn_out_w, gdn_out_n);
                } }
                match &l.mlp {
                    Ffn::Moe(m) => {
                        if wide("expert") { add(&m.gate_up, &[format!("{lp}.mlp.experts.gate_up_proj")], true, false); add(&m.down, &[format!("{lp}.mlp.experts.down_proj")], true, false); }
                        let (mlp_w, mlp_n) = (wide("mlp"), narrow("mlp"));
                        if mlp_w || mlp_n {
                            let (sg, su, sd) = (format!("{lp}.mlp.shared_expert.gate_proj"), format!("{lp}.mlp.shared_expert.up_proj"), format!("{lp}.mlp.shared_expert.down_proj"));
                            add(&m.shared.gate, &[sg.clone()], mlp_w, mlp_n); add(&m.shared.up, &[su.clone()], mlp_w, mlp_n); add(&m.shared.down, &[sd], mlp_w, mlp_n);
                            if let Some(w) = &m.shared_gate_up { add(w, &[sg, su], mlp_w, mlp_n); }
                        }
                    }
                    Ffn::Dense(m) => { let (mw, mn) = (wide("mlp"), narrow("mlp")); if mw || mn { add(&m.gate, &[format!("{lp}.mlp.gate_proj")], mw, mn); add(&m.up, &[format!("{lp}.mlp.up_proj")], mw, mn); add(&m.down, &[format!("{lp}.mlp.down_proj")], mw, mn); } }
                }
                let (hc_w, hc_n) = (wide("hc"), narrow("hc"));
                if hc_w || hc_n { if let Some((a, b)) = &l.hc {
                    add(&a.down, &[format!("{lp}.attn_hyper_connection.input_mix_weight_down")], hc_w, hc_n); add(&a.up, &[format!("{lp}.attn_hyper_connection.input_mix_weight_up")], hc_w, hc_n);
                    add(&b.down, &[format!("{lp}.mlp_hyper_connection.input_mix_weight_down")], hc_w, hc_n); add(&b.up, &[format!("{lp}.mlp_hyper_connection.input_mix_weight_up")], hc_w, hc_n);
                } }
                let (ple_w, ple_n) = (wide("ple"), narrow("ple"));
                if ple_w || ple_n { if let Some(p) = &l.ple { add(&p.key_proj, &[format!("{lp}.ple.key_proj")], ple_w, ple_n); add(&p.value_proj, &[format!("{lp}.ple.value_proj")], ple_w, ple_n); } }
            }
            let (lm_w, lm_n) = (wide("lmhead"), narrow("lmhead") || (wide("lmhead") && narrow_requested));
            if lm_w || lm_n { if let Some(w) = &self.lm_head {
                add(w, &["lm_head".to_string()], lm_w, lm_n);
            } }
        }
        if enabled.is_empty() && narrow_enabled.is_empty() { println!("W4A4 runtime: no NVFP4 weight in groups {groups:?} — off"); return Ok(()); }
        println!("W4A4 policy: prefill={prefill_groups:?}, decode+verify={verify_groups:?}");
        let rows_max = crate::batch::PREFILL_CHUNK.max(self.moe_g_pf.ppad_max);
        let tiles_max = self.moe_g_pf.ppad_max / crate::w4a4::W4_BN + self.cfg.num_experts + 2;
        self.w4a4 = Some(crate::w4a4::W4a4State::build(&self.dev, groups, enabled, narrow_enabled, x_gs, rows_max, k_max, tiles_max)?);
        Ok(())
    }

    /// Dense W4A4 prefill GEMM: x [batch][inn] row-major bf16 -> out [batch][outn], sub-batched by
    /// PREFILL_CHUNK rows through the packed-activation scratch.
    pub(super) fn w4a4_dense(&self, w4: &crate::w4a4::W4a4State, w: &W, x: &B, out: &mut B,
                  inn: usize, outn: usize, batch: usize, x_gs: f32) {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let W::Nvfp4 { qweight, scales, gs, .. } = w else { unreachable!() };
        let (wq, ws, gs) = (*qweight.device_ptr() as u64, *scales.device_ptr() as u64, *gs.device_ptr() as u64);
        assert!(inn <= w4.k_max, "W4A4: K {inn} exceeds the scratch K {}", w4.k_max);
        const SUB: usize = crate::batch::PREFILL_CHUNK;
        let (bq, sb) = (*w4.bq.device_ptr() as u64, *w4.sb.device_ptr() as u64);
        let (xq, oq) = (*x.device_ptr() as u64, *out.device_ptr() as u64);
        let mut off = 0usize;
        while off < batch {
            let rows = SUB.min(batch - off);
            let pad8 = rows.div_ceil(8) * 8;
            unsafe {
                w4.quant.clone().launch_on_stream(&self.stream,
                    LaunchConfig { grid_dim: ((inn / 64) as u32, (pad8 / 8) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (xq + (off * inn * 2) as u64, inn as i32, rows as i32, pad8 as i32, bq, sb, x_gs))
                    .expect("w4a4_quant_pack_b launch");
                if rows <= MAX_VERIFY && crate::w4a4::n8_on() {
                    w4.gemm_n8.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (outn.div_ceil(128) as u32, rows.div_ceil(8) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: crate::w4a4::W4_N8_SMEM },
                        (wq, ws, bq, sb, gs, oq + (off * outn * 2) as u64, outn as i32, rows as i32, inn as i32, 1.0f32 / x_gs))
                        .expect("w4a4_gemm_n8_b launch");
                } else {
                    w4.gemm.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (crate::w4a4::W4a4State::dense_grid(outn, rows), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: crate::w4a4::W4_SMEM },
                        (wq, ws, bq, sb, gs, oq + (off * outn * 2) as u64, outn as i32, rows as i32, inn as i32, 1.0f32 / x_gs))
                        .expect("w4a4_gemm_b launch");
                }
            }
            off += rows;
        }
        if w4.trace { eprintln!("[w4a4] dense outn={outn} inn={inn} batch={batch} x_gs={x_gs}"); }
        if crate::w4a4::check_on() {
            // --w4a4-check: recompute through the bf16 chain (dequant + cuBLAS) and compare per row.
            let mut r = unsafe { self.dev.alloc::<half::bf16>(out.len()).unwrap() };
            let xf = self.w4a4_fakequant(w4, x, batch, inn, x_gs);
            self.gemm_quant_prefill_slow(w, &xf, &mut r, inn, outn, batch);
            self.w4a4_compare(out, &r, batch, outn, &format!("dense outn={outn} inn={inn}"));
        }
    }

    /// --w4a4-check helper: Y = dequant(quant(X)) with the packer's recipe, [rows][k] bf16.
    pub(super) fn w4a4_fakequant(&self, w4: &crate::w4a4::W4a4State, x: &B, rows: usize, k: usize, x_gs: f32) -> B {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let y = self.dev.alloc_zeros::<half::bf16>(rows * k).unwrap();
        unsafe {
            w4.fakequant.clone().launch_on_stream(&self.stream,
                LaunchConfig { grid_dim: ((k / 64) as u32, rows.div_ceil(8) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (*x.device_ptr() as u64, *y.device_ptr() as u64, k as i32, rows as i32, x_gs)).expect("w4a4_fakequant_b launch");
        }
        y
    }

    /// Per-row relative L2 of `got` vs `want` ([rows][cols] bf16, row-major) over the first `rows`
    /// rows; rows whose reference is all-zero (padding) are skipped.
    pub(super) fn w4a4_compare(&self, got: &B, want: &B, rows: usize, cols: usize, what: &str) {
        self.sync_stream();
        let a: Vec<half::bf16> = unsafe { self.dev.dtoh_sync_copy(got).unwrap() };
        let b: Vec<half::bf16> = unsafe { self.dev.dtoh_sync_copy(want).unwrap() };
        let (mut sum, mut n, mut mx, mut bad, mut first_bad) = (0f64, 0usize, 0f64, 0usize, usize::MAX);
        for r in 0..rows {
            let (mut num, mut den) = (0f64, 0f64);
            for c in 0..cols { let x = a[r * cols + c].to_f32() as f64; let y = b[r * cols + c].to_f32() as f64; num += (x - y) * (x - y); den += y * y; }
            if den == 0.0 { continue; }
            let rl = (num / den).sqrt();
            if !rl.is_finite() || rl > 0.05 { bad += 1; if first_bad == usize::MAX { first_bad = r; } }
            sum += rl; n += 1; if rl > mx { mx = rl; }
        }
        eprintln!("[w4a4-check] {what} rows={n}: relL2 mean {:.3e} max {:.3e} bad(>0.05) {bad}{}",
                  if n > 0 { sum / n as f64 } else { 0.0 }, mx, if bad > 0 { format!(" first at row {first_bad}") } else { String::new() });
    }

    /// Grouped-prefill MoE W4A4 GEMM over the permuted rows: `x` [rows][k] (x_perm or h_p, rows =
    /// ppad_cap), stacked expert weights (M rows per expert), out [rows][m]. The 128-row tile map
    /// (per expert region of `g.poff`) is built on device on the gate_up call and reused for down.
    pub(super) fn w4a4_moe(&self, w4: &crate::w4a4::W4a4State, x: &B, rows: usize, k: usize, wq: u64, ws: u64, gs: u64,
                m: usize, out: &B, g: &MoeGroupedScratch, ne: usize, e_base: i32, build_map: bool, x_gs: f32) {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let pad8 = rows.div_ceil(8) * 8;   // the packer works in 8-row groups (rows >= `rows` zero-filled)
        assert!(k <= w4.k_max && pad8 <= w4.rows_max, "W4A4 MoE: rows {rows} x K {k} exceed the scratch ({} x {})", w4.rows_max, w4.k_max);
        assert!(k % 64 == 0);
        let (bq, sb) = (*w4.bq.device_ptr() as u64, *w4.sb.device_ptr() as u64);
        unsafe {
            w4.quant.clone().launch_on_stream(&self.stream,
                LaunchConfig { grid_dim: ((k / 64) as u32, (pad8 / 8) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (*x.device_ptr() as u64, k as i32, rows as i32, pad8 as i32, bq, sb, x_gs))
                .expect("w4a4_quant_pack_b (moe) launch");
            if build_map {
                w4.tilemap.clone().launch_on_stream(&self.stream,
                    LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                    (*w4.tmap.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32, w4.tiles_max as i32))
                    .expect("w4a4_moe_tilemap_b launch");
            }
            w4.gemm_moe.clone().launch_on_stream(&self.stream,
                LaunchConfig { grid_dim: (w4.tiles_max as u32, m.div_ceil(128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: crate::w4a4::W4_SMEM },
                (*out.device_ptr() as u64, wq, ws, gs, bq, sb, *w4.tmap.device_ptr() as u64,
                 *g.poff.device_ptr() as u64, m as i32, k as i32, e_base, 1.0f32 / x_gs))
                .expect("w4a4_gemm_moe_b launch");
        }
        if w4.trace { eprintln!("[w4a4] moe m={m} k={k} rows={rows} x_gs={x_gs}"); }
    }
}


// ---------------------------------------------------------------- --calib-igs: activation amax per NVFP4 weight
impl GpuModel {
    /// Every NVFP4 2-D weight of the trunk (+ lm_head) as (tensor stem, qweight device pointer, K).
    /// A fused tensor (qkv, GDN in_proj, shared gate|up) lists one entry per source stem, all with
    /// the fused pointer — the same activation feeds them.
    pub fn nvfp4_weights_by_name(&self) -> Vec<(String, u64, usize)> {
        let lm = "model.language_model";
        let mut v: Vec<(String, u64, usize)> = Vec::new();
        let mut add = |w: &W, names: &[String]| { if let W::Nvfp4 { qweight, k, .. } = w { for n in names { v.push((n.clone(), *qweight.device_ptr() as u64, *k)); } } };
        for (li, l) in self.layers.iter().enumerate() {
            let lp = format!("{lm}.layers.{li}");
            if let Some(fa) = &l.fa {
                let (q, k, vv) = (format!("{lp}.self_attn.q_proj"), format!("{lp}.self_attn.k_proj"), format!("{lp}.self_attn.v_proj"));
                match &fa.qkv { AttnIn::Fused(w) => add(w, &[q, k, vv]), AttnIn::Split { q: wq, k: wk, v: wv } => { add(wq, &[q]); add(wk, &[k]); add(wv, &[vv]); } }
                add(&fa.o_proj, &[format!("{lp}.self_attn.o_proj")]);
                if let Some(ix) = &fa.indexer { add(&ix.qk_proj, &[format!("{lp}.self_attn.indexer.index_qk_proj")]); }
            }
            if let Some(la) = &l.la {
                let n = |s: &str| format!("{lp}.linear_attn.{s}");
                match &la.in_proj {
                    GdnIn::Fused(w) => add(w, &[n("in_proj_qkv"), n("in_proj_z"), n("in_proj_b"), n("in_proj_a")]),
                    GdnIn::Split { qkv, z, b, a } => { add(qkv, &[n("in_proj_qkv")]); add(z, &[n("in_proj_z")]); add(b, &[n("in_proj_b")]); add(a, &[n("in_proj_a")]); }
                }
                add(&la.out_proj, &[n("out_proj")]);
            }
            match &l.mlp {
                Ffn::Moe(m) => {
                    add(&m.gate_up, &[format!("{lp}.mlp.experts.gate_up_proj")]); add(&m.down, &[format!("{lp}.mlp.experts.down_proj")]);
                    let (sg, su, sd) = (format!("{lp}.mlp.shared_expert.gate_proj"), format!("{lp}.mlp.shared_expert.up_proj"), format!("{lp}.mlp.shared_expert.down_proj"));
                    add(&m.shared.gate, &[sg.clone()]); add(&m.shared.up, &[su.clone()]); add(&m.shared.down, &[sd]);
                    if let Some(w) = &m.shared_gate_up { add(w, &[sg, su]); }
                    add(&m.router, &[format!("{lp}.mlp.gate")]);
                }
                Ffn::Dense(m) => { add(&m.gate, &[format!("{lp}.mlp.gate_proj")]); add(&m.up, &[format!("{lp}.mlp.up_proj")]); add(&m.down, &[format!("{lp}.mlp.down_proj")]); }
            }
            if let Some((a, b)) = &l.hc {
                add(&a.down, &[format!("{lp}.attn_hyper_connection.input_mix_weight_down")]); add(&a.up, &[format!("{lp}.attn_hyper_connection.input_mix_weight_up")]);
                add(&b.down, &[format!("{lp}.mlp_hyper_connection.input_mix_weight_down")]); add(&b.up, &[format!("{lp}.mlp_hyper_connection.input_mix_weight_up")]);
            }
            if let Some(p) = &l.ple { add(&p.key_proj, &[format!("{lp}.ple.key_proj")]); add(&p.value_proj, &[format!("{lp}.ple.value_proj")]); }
        }
        if let Some(w) = &self.lm_head { add(w, &["lm_head".to_string()]); }
        v
    }
    pub fn igs_arm(&self, ptrs: &[u64]) {
        let mut m = std::collections::HashMap::new();
        for &p in ptrs {
            m.entry(p).or_insert_with(|| IgsTapBuffers {
                stats: self.dev.htod_sync_copy(&vec![0u64; IGS_HIST_BINS + 2]).unwrap(),
                running_max: self.dev.htod_sync_copy(&[0u32]).unwrap(),
            });
        }
        *self.igs_tap.lock().unwrap() = Some(m);
    }
    /// Disarm and return mergeable per-block histograms and the literal running max.
    pub fn igs_disarm(&self) -> std::collections::HashMap<u64, IgsHistogram> {
        self.sync_stream();
        let m = self.igs_tap.lock().unwrap().take().unwrap_or_default();
        m.iter().map(|(p, tap)| {
            let raw = self.dev.dtoh_sync_copy(&tap.stats).unwrap();
            let max_bits = self.dev.dtoh_sync_copy(&tap.running_max).unwrap()[0];
            (*p, IgsHistogram {
                bins: raw[..IGS_HIST_BINS].to_vec(),
                running_max: f32::from_bits(max_bits),
                zero_blocks: raw[IGS_HIST_BINS],
                invalid_blocks: raw[IGS_HIST_BINS + 1],
            })
        }).collect()
    }
    /// Fold `n` bf16 values into the per-16 activation histogram for weight `ptr`.
    pub(super) fn igs_tap_x(&self, ptr: u64, x: &B, n: usize) {
        let g = self.igs_tap.lock().unwrap();
        let Some(t) = g.as_ref() else { return; };
        let Some(tap) = t.get(&ptr) else { return; };
        blaunch!(self, "igs_hist_b", grid(n.div_ceil(16)), (256,1,1), 0,
            (d(&tap.stats), d(&tap.running_max), d(x), n as i64));
    }
}
