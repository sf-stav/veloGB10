//! Identity lines (G-T1-a), the CPU / GPU slice oracle (`--probe-exl3-slice`: a sliced trellis tensor must
//! reconstruct exactly the matching slice of the full tensor) and the MoE ulp perturbation arm. See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Identity lines (G-T1-a)
// ---------------------------------------------------------------------------------------------

fn sha256_hex(path: &str) -> String {
    use sha2::{Digest, Sha256};
    match std::fs::read(path) {
        Ok(b) => { let d = Sha256::digest(&b); d.iter().map(|x| format!("{x:02x}")).collect() }
        Err(_) => "missing".to_string(),
    }
}

/// The per-step tripwire of the harness passes (`run_tp_xtp`): both ranks publish (step, argmax) and COMPARE.
/// S-B9-CF-AGREE16B (W6): the three passes used to test `agree_ext(..).is_none()` alone, which at TP=2 is
/// only "the link is up": `net_agree` hands back the PEER's token and nobody looked at it, so two ranks
/// with different argmaxes agreed step after step (the served path has compared since TP-E; at world > 2
/// the hub compares inside `agree_ext`). `tp_agree_eq` is that comparison; on a mismatch the link is
/// aborted so the peer fails at its next collective instead of at a 10 s timeout.
pub(super) fn probe_agree(step: u64, argmax: u32) -> std::result::Result<(), String> {
    crate::exl3_serve::tp_agree_eq(step, 0, 1, argmax).map_err(|why| { crate::net::abort_link(); why })
}

/// `TP-IDENT` — binary sha256 + the two PTX modules this path loads (embedded bytes since EMB1) + the build token.
pub fn identity_line(role: &str) -> String {
    let exe = std::env::current_exe().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    let short = |h: String| h.chars().take(16).collect::<String>();
    format!("TP-IDENT role={role} host={} bin={} exl3_bench.ptx={} gpu_batch.ptx={} kernel_build_id={} source_build_id={}",
            std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
            short(sha256_hex(&exe)),
            short(crate::exl3_tune::sha256_hex(crate::exl3_bench::bench_ptx_src().as_bytes())),
            short(crate::exl3_tune::sha256_hex(crate::ptx_embed::GPU_BATCH.as_bytes())),
            env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID"))
}

// ---------------------------------------------------------------------------------------------
// Rung 1 — G-T1-0 slice oracle
// ---------------------------------------------------------------------------------------------

fn le_i16v(b: &[u8]) -> Vec<i16> { b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect() }
fn le_u16v(b: &[u8]) -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

/// Gather the 16x16-block sub-array (row block ranges x col block ranges, in 16-units) of a
/// trellis [kb][nb][16*bits] — the load-time slice/re-tile (a copy, never a requant).
pub(in crate::exl3_forward) fn slice_trellis<T: Copy>(tr: &[T], nb: usize, k2: usize, rows: &[(usize, usize)], cols: &[(usize, usize)]) -> Vec<T> {
    let bw = 8 * k2; // uint16 words per tile (16*K, or 8*k2 with K possibly X.5)
    let mut out = Vec::new();
    for &(r0, r1) in rows {
        for r in r0..r1 {
            for &(c0, c1) in cols {
                out.extend_from_slice(&tr[(r * nb + c0) * bw..(r * nb + c1) * bw]);
            }
        }
    }
    out
}

fn span_len(v: &[(usize, usize)]) -> usize { v.iter().map(|(a, b)| b - a).sum() }

/// Blocked normalized Walsh–Hadamard over 128-element groups (the operator's H128; any fixed
/// 128-point orthonormal transform exposes a cut block identically).
fn had128_blocks(v: &mut [f64]) {
    let s = 1.0 / (128f64).sqrt();
    for blk in v.chunks_exact_mut(128) {
        let mut len = 1;
        while len < 128 {
            for i in (0..128).step_by(2 * len) {
                for j in i..i + len {
                    let (a, b) = (blk[j], blk[j + len]);
                    blk[j] = a + b;
                    blk[j + len] = a - b;
                }
            }
            len <<= 1;
        }
        for x in blk.iter_mut() { *x *= s; }
    }
}

/// y = svh ⊙ H_N( W^T · H_K(x ⊙ suh) ) in f64 (W = decoded f16 bits, row-major [K][N]).
fn op_f64(w: &[u16], k: usize, n: usize, suh: &[u16], svh: &[u16], x: &[f64]) -> Vec<f64> {
    let mut xs: Vec<f64> = (0..k).map(|i| x[i] * f16::from_bits(suh[i]).to_f64()).collect();
    had128_blocks(&mut xs);
    let mut z = vec![0f64; n];
    for i in 0..k {
        let xi = xs[i];
        if xi == 0.0 { continue; }
        let row = &w[i * n..(i + 1) * n];
        for (zj, &wb) in z.iter_mut().zip(row) { *zj += xi * f16::from_bits(wb).to_f64(); }
    }
    had128_blocks(&mut z);
    for j in 0..n { z[j] *= f16::from_bits(svh[j]).to_f64(); }
    z
}

/// Row-parallel partial: z_r = W[rows]^T · H(x[rows] ⊙ suh[rows]) (pre output-transform).
fn op_rows_partial(w: &[u16], n: usize, suh: &[u16], x: &[f64], rows: &[(usize, usize)]) -> Vec<f64> {
    let idx: Vec<usize> = rows.iter().flat_map(|&(a, b)| (a * 16)..(b * 16)).collect();
    let mut xs: Vec<f64> = idx.iter().map(|&i| x[i] * f16::from_bits(suh[i]).to_f64()).collect();
    had128_blocks(&mut xs);
    let mut z = vec![0f64; n];
    for (p, &i) in idx.iter().enumerate() {
        let row = &w[i * n..(i + 1) * n];
        for (zj, &wb) in z.iter_mut().zip(row) { *zj += xs[p] * f16::from_bits(wb).to_f64(); }
    }
    z
}

fn rel_l2_f64(a: &[f64], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { num += (x - y) * (x - y); den += y * y; }
    (num / den.max(1e-300)).sqrt()
}

/// Split plans for one tensor class at world W (ranges in 16-row/col block units).
/// `Col`/`Row` = contiguous 128-aligned ranges per rank; `Retile` = a per-rank column SET.
enum SplitPlan {
    Col(Vec<Vec<(usize, usize)>>),
    Row(Vec<Vec<(usize, usize)>>),
}

/// Even 128-aligned split of `dim` into W parts if possible, else the uneven 128-block deal
/// (ceil-first), else None (dim not a multiple of 128). Returns (ranges per rank, even?).
pub(super) fn split_128(dim: usize, w: usize) -> Option<(Vec<Vec<(usize, usize)>>, bool)> {
    if dim % 128 != 0 { return None; }
    let blocks = dim / 128;
    if blocks < w { return None; }
    let even = blocks % w == 0;
    let mut out = Vec::new();
    let mut b0 = 0;
    for r in 0..w {
        let nb = blocks / w + usize::from(r < blocks % w);
        out.push(vec![(b0 * 8, (b0 + nb) * 8)]);
        b0 += nb;
    }
    Some((out, even))
}

/// The design §1.3 per-class map at world W: the column set / row range each rank takes.
fn planned_split(name: &str, k: usize, n: usize, w: usize, cfg: &crate::qwen::Config) -> Option<(String, SplitPlan)> {
    let s16 = |a: usize, b: usize| (a / 16, b / 16);
    if name.ends_with("linear_attn.in_proj_qkv") {
        // [q nk×kd | k nk×kd | v nv×vd], contiguous k<->v head groups: rank r takes the r-th
        // 1/W of each section (a load-time re-tile — a copy of 128-aligned block columns).
        let (nk, kd, nv, vd) = (cfg.lin_num_k_heads, cfg.lin_k_dim, cfg.lin_num_v_heads, cfg.lin_v_dim);
        if nk % w != 0 || nv % w != 0 || n != 2 * nk * kd + nv * vd { return None; }
        let qs = nk * kd / w;
        let vs = nv * vd / w;
        let plans = (0..w).map(|r| vec![s16(r * qs, (r + 1) * qs), s16(nk * kd + r * qs, nk * kd + (r + 1) * qs),
                                         s16(2 * nk * kd + r * vs, 2 * nk * kd + (r + 1) * vs)]).collect();
        return Some((format!("retile q|k|v {}+{}+{} cols/rank", qs, qs, vs), SplitPlan::Col(plans)));
    }
    if name.ends_with("self_attn.k_proj") || name.ends_with("self_attn.v_proj") {
        // one KV head per rank at W=2; at W=4 KV head r/2 (pairs replicate)
        let hd = cfg.head_dim;
        let nkv = cfg.num_kv_heads;
        if n != nkv * hd { return None; }
        let plans = (0..w).map(|r| { let hh = r * nkv / w; vec![s16(hh * hd, (hh + 1) * hd)] }).collect();
        return Some((format!("col, KV head r*{nkv}/W ({hd} cols)"), SplitPlan::Col(plans)));
    }
    let col = name.ends_with("self_attn.q_proj") || name.ends_with("linear_attn.in_proj_z")
        || name.ends_with("shared_expert.gate_proj") || name.ends_with("shared_expert.up_proj")
        || name == "lm_head" || name.ends_with("gate_proj") || name.ends_with("up_proj")
        || name.ends_with("index_qk_proj") || name.starts_with("mtp.fc_");
    let row = name.ends_with("o_proj") || name.ends_with("out_proj") || name.ends_with("down_proj");
    if col {
        let (p, even) = split_128(n, w)?;
        return Some((format!("col {}", if even { "even" } else { "UNEVEN 128-block" }), SplitPlan::Col(p)));
    }
    if row {
        let (p, even) = split_128(k, w)?;
        return Some((format!("row {}", if even { "even" } else { "UNEVEN 128-block" }), SplitPlan::Row(p)));
    }
    None
}

fn class_of(name: &str) -> String {
    if name == "lm_head" { return "lm_head".into(); }
    if name.starts_with("mtp.fc_") { return format!("mtp.{}", name.trim_start_matches("mtp.")); }
    let pre = if name.starts_with("mtp.") { "mtp." } else { "" };
    let tail = name.rsplit('.').next().unwrap_or(name);
    if name.contains(".experts.") { return format!("{pre}expert.{tail}"); }
    if name.contains("shared_expert.") { return format!("{pre}shared_expert.{tail}"); }
    if name.contains(".indexer.") { return format!("{pre}indexer.{tail}"); }
    if name.contains("linear_attn.") { return format!("{pre}gdn.{tail}"); }
    if name.contains("self_attn.") { return format!("{pre}attn.{tail}"); }
    format!("{pre}{tail}")
}

/// `--probe-exl3-slice`: G-T1-0. For every quantized tensor class (every non-expert module of
/// every layer + a sample of experts), at W = 2 and 4:
///   (a) decode(slice(trellis)) == slice(decode(trellis)) bit-exact over the planned split
///       (design §1.3) — the open item of §1.2 (does a block's decode depend on its GLOBAL index?);
///   (b) the f64 operator on the sliced tensor + suh/svh slices == the full operator's slice
///       (col: exactly; row: the partials' sum == full, pre-svh reassociation only);
///   (c) negative control: a 64-offset (non-128) column cut must NOT match (a cut H128 block).
/// `gpu` adds the one GPU check (served decode + m=1 chain on sliced vs full Quads).
pub fn probe_slice(dir: &str, experts_per_layer: usize, gpu: bool) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let cfg = pack.cfg.clone();
    let t0 = std::time::Instant::now();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0 };
    // (class) -> [modules, blocks checked, bad values, op checks, max col op err, max row op rel err, neg-control ok]
    let mut per: BTreeMap<String, (usize, usize, usize, usize, f64, f64, usize, String)> = BTreeMap::new();
    let mut plain_report: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_bad = 0usize;
    let mut neg_fail = 0usize;
    let pick_expert = |name: &str| -> bool {
        // experts.<e>.  keep e < experts_per_layer/2 or e >= ne - experts_per_layer/2 (both ranks' ends)
        if let Some(i) = name.find(".experts.") {
            let rest = &name[i + 9..];
            let e: usize = rest.split('.').next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let half = (experts_per_layer / 2).max(1);
            return e < half || e >= cfg.num_experts.saturating_sub(half);
        }
        true
    };
    let mut n_mod = 0usize;
    for m in pack.modules.iter() {
        if !pick_expert(&m.name) { continue; }
        n_mod += 1;
        let tr = le_i16v(&m.trellis.read_bytes(dir)?);
        let suh = le_u16v(&m.suh.read_bytes(dir)?);
        let svh = le_u16v(&m.svh.read_bytes(dir)?);
        let (k, n, k2) = (m.k, m.n, m.k2);
        let (kb, nb) = (k / 16, n / 16);
        anyhow::ensure!(tr.len() == kb * nb * 8 * k2 && suh.len() == k && svh.len() == n,
            "{}: trellis/suh/svh shapes do not match k={k} n={n} k2={k2}", m.name);
        let full = crate::exl3::decode_trellis_host(&tr, kb, nb, k2);
        let class = class_of(&m.name);
        let x: Vec<f64> = (0..k).map(|_| next()).collect();
        // the full operator once per module (skipped for the 635M-weight lm_head beyond one check)
        let y_full = op_f64(&full, k, n, &suh, &svh, &x);
        let ent = per.entry(class.clone()).or_insert((0, 0, 0, 0, 0.0, 0.0, 0, String::new()));
        ent.0 += 1;
        for &w in &[2usize, 4] {
            let Some((desc, plan)) = planned_split(&m.name, k, n, w, &cfg) else {
                if ent.7.is_empty() {
                    ent.7 = if m.name.starts_with("model.language_model.") || m.name.starts_with("mtp.") || m.name == "lm_head" {
                        format!("W={w}: NO 128-aligned split of k={k} n={n} — UNSPLITTABLE")
                    } else {
                        format!("not in the TP map (outside the text engine; k={k} n={n}) — replicated/unused")
                    };
                }
                continue;
            };
            if !ent.7.contains(&format!("W={w}:")) { ent.7 += &format!(" W={w}:{desc}"); }
            match &plan {
                SplitPlan::Col(parts) => {
                    for cols in parts {
                        let s = slice_trellis(&tr, nb, k2, &[(0, kb)], cols);
                        let ns = span_len(cols);
                        let dec = crate::exl3::decode_trellis_host(&s, kb, ns, k2);
                        let cidx: Vec<usize> = cols.iter().flat_map(|&(a, b)| (a * 16)..(b * 16)).collect();
                        let mut bad = 0usize;
                        for r in 0..k {
                            for (p, &c) in cidx.iter().enumerate() {
                                if dec[r * ns * 16 + p] != full[r * n + c] { bad += 1; }
                            }
                        }
                        ent.1 += kb * ns; ent.2 += bad; total_bad += bad;
                        let svh_s: Vec<u16> = cidx.iter().map(|&c| svh[c]).collect();
                        let y_s = op_f64(&dec, k, ns * 16, &suh, &svh_s, &x);
                        let err = cidx.iter().enumerate().map(|(p, &c)| (y_s[p] - y_full[c]).abs()).fold(0f64, f64::max);
                        ent.3 += 1; ent.4 = ent.4.max(err);
                    }
                }
                SplitPlan::Row(parts) => {
                    let mut zsum = vec![0f64; n];
                    for rows in parts {
                        let s = slice_trellis(&tr, nb, k2, rows, &[(0, nb)]);
                        let ks = span_len(rows);
                        let dec = crate::exl3::decode_trellis_host(&s, ks, nb, k2);
                        let ridx: Vec<usize> = rows.iter().flat_map(|&(a, b)| (a * 16)..(b * 16)).collect();
                        let mut bad = 0usize;
                        for (p, &r) in ridx.iter().enumerate() {
                            if dec[p * n..(p + 1) * n] != full[r * n..(r + 1) * n] {
                                bad += dec[p * n..(p + 1) * n].iter().zip(&full[r * n..(r + 1) * n]).filter(|(a, b)| a != b).count();
                            }
                        }
                        ent.1 += ks * nb; ent.2 += bad; total_bad += bad;
                        // this rank's partial from ITS slices only (suh slice + its x rows)
                        let suh_s: Vec<u16> = ridx.iter().map(|&r| suh[r]).collect();
                        let x_s: Vec<f64> = ridx.iter().map(|&r| x[r]).collect();
                        let z = op_rows_partial(&dec, n, &suh_s, &x_s, &[(0, ks)]);
                        for (a, b) in zsum.iter_mut().zip(&z) { *a += b; }
                    }
                    had128_blocks(&mut zsum);
                    for j in 0..n { zsum[j] *= f16::from_bits(svh[j]).to_f64(); }
                    let e = rel_l2_f64(&zsum, &y_full);
                    ent.3 += 1; ent.5 = ent.5.max(e);
                }
            }
        }
        // (c) negative control on the column axis: a 64-offset cut [64, 64+128) — the decoded
        // weights are block-exact at any 16 multiple, but the OPERATOR's output H128 is cut.
        if n >= 256 {
            let cols = [(4usize, 12usize)];
            let s = slice_trellis(&tr, nb, k2, &[(0, kb)], &cols);
            let dec = crate::exl3::decode_trellis_host(&s, kb, 8, k2);
            let svh_s: Vec<u16> = (64..192).map(|c| svh[c]).collect();
            let y_s = op_f64(&dec, k, 128, &suh, &svh_s, &x);
            let err = (0..128).map(|p| (y_s[p] - y_full[64 + p]).abs()).fold(0f64, f64::max);
            let scale = y_full[64..192].iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-30);
            if err / scale > 1e-6 { ent.6 += 1; } else { neg_fail += 1; }
        }
        if n_mod % 200 == 0 {
            crate::dprintln!("  slice oracle: {n_mod} modules ({:.0}s)", t0.elapsed().as_secs_f32());
        }
    }
    // plain (non-trellis) tensors the TP map touches: fp16/bf16 as stored — slicing is a copy
    let mut plain_names: Vec<(String, String)> = Vec::new();
    for sh in &pack.shards {
        for (k, v) in read_hdr(&format!("{}/{}", dir.trim_end_matches('/'), sh))? { plain_names.push((k, v.dtype.clone())); }
    }
    for (cls, pat) in [("gdn.in_proj_a|b", "linear_attn.in_proj_"), ("hc mixers", "hyper_connection"),
                       ("router gate", "mlp.gate.weight"), ("shared_expert_gate", "shared_expert_gate.weight")] {
        let c = plain_names.iter().filter(|(k, _)| k.contains(pat) && !k.contains(".trellis") && !k.contains(".suh")
                                                && !k.contains(".svh") && !k.contains(".mul1")).count();
        let dts: std::collections::BTreeSet<&str> = plain_names.iter().filter(|(k, _)| k.contains(pat) && !k.contains(".trellis")
                                                && !k.contains(".suh") && !k.contains(".svh") && !k.contains(".mul1"))
                                                .map(|(_, d)| d.as_str()).collect();
        plain_report.insert(format!("{cls} ({})", dts.into_iter().collect::<Vec<_>>().join("/")), c);
    }
    crate::dprintln!("SLICE_ORACLE per class (W = 2 and 4, planned design §1.3 splits):");
    for (cls, (mods, blocks, bad, ops, ecol, erow, neg, desc)) in &per {
        crate::dprintln!("SLICE_CLASS {cls:<28} modules {mods:>5} blocks {blocks:>10} bad_values {bad} op_checks {ops} \
                  col_op_maxabs {ecol:.3e} row_op_relL2 {erow:.3e} neg_ctrl_detected {neg}/{mods} |{desc}");
    }
    for (cls, c) in &plain_report {
        crate::dprintln!("SLICE_PLAIN {cls:<34} tensors {c:>5}: not trellis-coded — a slice is a plain copy (exact by construction)");
    }
    let ok = total_bad == 0 && neg_fail == 0 && !per.is_empty();
    crate::dprintln!("SLICE_ORACLE_VERDICT {} modules={n_mod} bad_values={total_bad} neg_control_misses={neg_fail} wall={:.1}s",
             if ok { "SLICE_OK" } else { "SLICE_FAIL" }, t0.elapsed().as_secs_f32());
    if gpu { probe_slice_gpu(dir, &pack)?; }
    if !ok { anyhow::bail!("slice oracle FAILED"); }
    Ok(())
}

fn quad_from_host(dev: &Arc<CudaDevice>, tr: &[i16], suh: &[u16], svh: &[u16], k: usize, n: usize, k2: usize) -> Result<Quad> {
    let tru: Vec<u16> = tr.iter().map(|&v| v as u16).collect();
    Ok(Quad {
        tr: dev.htod_sync_copy(&tru)?,
        suh: dev.htod_sync_copy(suh)?,
        svh: dev.htod_sync_copy(svh)?,
        bits: (k2 / 2) as i32,
        k2: k2 as i32,
        k: k as i32,
        n: n as i32,
        ks: chain_ks(k as i32, n as i32),
        lmh: false,
        cnt: dev.htod_sync_copy(&vec![0u32; (n / 128).max(1)])?,
        rowpar: false,
        fx: None,
    })
}

/// The one GPU check: for one module of each dense class (+ one expert and the lm_head), the
/// served decode (exl3_reconstruct) of each rank's sliced tensor == the CPU decode's slice
/// (bitwise), and the served m=1 chain on the sliced Quad vs the full chain's slice (bitwise for
/// a column slice with the same ks; relL2 otherwise — a per-rank shape may pick a different ks).
fn probe_slice_gpu(dir: &str, pack: &Exl3Pack) -> Result<()> {
    tune::ensure_frozen("probe-exl3-slice");
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::exl3_bench::load_module_pub(&dev)?;
    let s = fork_blocking_stream(&dev); // AGENTS §2.1: blocking stream
    let l = Launcher { dev: &dev, stream: &s };
    let rc = dev.htod_sync_copy(&crate::exl3_bench::tmap_inv_table())?;
    let cfg = &pack.cfg;
    let wanted = ["model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.self_attn.o_proj",
                  "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.out_proj",
                  "model.language_model.layers.0.mlp.shared_expert.gate_proj", "model.language_model.layers.0.mlp.shared_expert.down_proj",
                  "model.language_model.layers.0.mlp.experts.7.gate_proj", "lm_head"];
    let mut all_ok = true;
    for name in wanted {
        let Some(m) = pack.modules.iter().find(|m| m.name == name) else { crate::dprintln!("SLICE_GPU {name}: not in pack"); continue };
        let tr = le_i16v(&m.trellis.read_bytes(dir)?);
        let suh = le_u16v(&m.suh.read_bytes(dir)?);
        let svh = le_u16v(&m.svh.read_bytes(dir)?);
        let (k, n, k2) = (m.k, m.n, m.k2);
        let (kb, nb) = (k / 16, n / 16);
        let full_cpu = crate::exl3::decode_trellis_host(&tr, kb, nb, k2);
        let Some((desc, plan)) = planned_split(name, k, n, 2, cfg) else { continue };
        let parts = match &plan { SplitPlan::Col(p) => p.clone(), SplitPlan::Row(p) => p.clone() };
        let is_col = matches!(plan, SplitPlan::Col(_));
        // full chain at m=1 (only for K == hidden inputs; sc-free buffers sized here)
        let qf = quad_from_host(&dev, &tr, &suh, &svh, k, n, k2)?;
        let xh_host: Vec<u16> = (0..k).map(|i| f16::from_f32(((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.2).to_bits()).collect();
        let x = dev.htod_sync_copy(&xh_host)?;
        let mut xh = dev.alloc_zeros::<u16>(k.max(16) * 16)?;
        let mut yraw = dev.alloc_zeros::<u16>(n * 16)?;
        let mut yf = dev.alloc_zeros::<u16>(n * 16)?;
        let ws = dev.alloc_zeros::<f32>(2_000_000)?; // == Scratch::chain_ws
        exl3_chain(&l, &qf, &x, &mut xh, &mut yraw, &mut yf, 1, &ws)?;
        dev.synchronize()?;
        let y_full = dev.dtoh_sync_copy(&yf)?;
        let mut dec_bad = 0usize;
        let mut chain_line = String::new();
        let mut row_sum = vec![0f32; n];
        for (r, rng) in parts.iter().enumerate() {
            let (rows, cols) = if is_col { (vec![(0, kb)], rng.clone()) } else { (rng.clone(), vec![(0, nb)]) };
            let st = slice_trellis(&tr, nb, k2, &rows, &cols);
            let (ks_, ns_) = (span_len(&rows), span_len(&cols));
            let ridx: Vec<usize> = rows.iter().flat_map(|&(a, b)| (a * 16)..(b * 16)).collect();
            let cidx: Vec<usize> = cols.iter().flat_map(|&(a, b)| (a * 16)..(b * 16)).collect();
            let suh_s: Vec<u16> = ridx.iter().map(|&i| suh[i]).collect();
            let svh_s: Vec<u16> = cidx.iter().map(|&i| svh[i]).collect();
            let q = quad_from_host(&dev, &st, &suh_s, &svh_s, ks_ * 16, ns_ * 16, k2)?;
            let wdec = dev.alloc_zeros::<u16>(ks_ * 16 * ns_ * 16)?;
            exl3_reconstruct(&l, *q.tr.device_ptr() as u64, *rc.device_ptr() as u64, *wdec.device_ptr() as u64,
                             (ks_ * 16) as i32, (ns_ * 16) as i32, k2 as i32)?;
            dev.synchronize()?;
            let wd = dev.dtoh_sync_copy(&wdec)?;
            for (pr, &rr) in ridx.iter().enumerate() {
                for (pc, &cc) in cidx.iter().enumerate() {
                    if wd[pr * ns_ * 16 + pc] != full_cpu[rr * n + cc] { dec_bad += 1; }
                }
            }
            // the served chain on this rank's slice
            let xs_host: Vec<u16> = ridx.iter().map(|&i| xh_host[i]).collect();
            let xs = dev.htod_sync_copy(&xs_host)?;
            let mut ys = dev.alloc_zeros::<u16>(ns_ * 16 * 16)?;
            let mut yraws = dev.alloc_zeros::<u16>(ns_ * 16 * 16)?;
            let mut xhs = dev.alloc_zeros::<u16>(ks_ * 16 * 16)?;
            let wss = dev.alloc_zeros::<f32>(2_000_000)?;
            exl3_chain(&l, &q, &xs, &mut xhs, &mut yraws, &mut ys, 1, &wss)?;
            dev.synchronize()?;
            let yh = dev.dtoh_sync_copy(&ys)?;
            if is_col {
                let neq = cidx.iter().enumerate().filter(|(p, &c)| yh[*p] != y_full[c]).count();
                let (mut num, mut den) = (0f64, 0f64);
                for (p, &c) in cidx.iter().enumerate() {
                    let (a, b) = (f16::from_bits(yh[p]).to_f64(), f16::from_bits(y_full[c]).to_f64());
                    num += (a - b) * (a - b); den += b * b;
                }
                chain_line += &format!(" r{r}[ks {}/{} bitneq {neq}/{} relL2 {:.2e}]", q.ks, qf.ks, cidx.len(), (num / den.max(1e-30)).sqrt());
            } else {
                for j in 0..n { row_sum[j] += f16::from_bits(yh[j]).to_f32(); }
            }
        }
        if !is_col {
            let (mut num, mut den) = (0f64, 0f64);
            for j in 0..n {
                let b = f16::from_bits(y_full[j]).to_f64();
                num += (row_sum[j] as f64 - b).powi(2); den += b * b;
            }
            chain_line = format!(" sum-of-partials vs full chain relL2 {:.2e} (f16 partials, reassociated)", (num / den.max(1e-30)).sqrt());
        }
        all_ok &= dec_bad == 0;
        crate::dprintln!("SLICE_GPU {name} [{desc}] served-decode bad_values {dec_bad} | m=1 chain:{chain_line}");
    }
    crate::dprintln!("SLICE_GPU_VERDICT {}", if all_ok { "SLICE_GPU_OK" } else { "SLICE_GPU_FAIL" });
    if !all_ok { anyhow::bail!("GPU slice check FAILED"); }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Rung 3 — the xtp probe
// ---------------------------------------------------------------------------------------------

/// Reduction-site ulp floor (xtp diagnostic, eager TP=1 only): while armed, every MoE call's
/// f16 output gets its mantissa LSB flipped in a pseudo-random ~half of its elements — a
/// one-f16-ulp perturbation AT the site TP=2 reassociates (the EP combine), the physically
/// matched companion of Q1's GDN-state bit-0 floor. 0 = off; else a running call counter.
pub(super) static MOE_ULP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(in crate::exl3_forward) fn moe_ulp_armed() -> bool { MOE_ULP.load(std::sync::atomic::Ordering::Relaxed) != 0 }

impl FwdModel {
    pub(in crate::exl3_forward) fn moe_ulp_perturb(&self, sc: &mut Scratch, m: usize) -> Result<()> {
        self.moe_ulp_perturb_buf(&mut sc.moe_out, m)
    }

    /// The same floor on any MoE output buffer (TP-B: the prefill's psc.moe_out, c rows).
    pub(in crate::exl3_forward) fn moe_ulp_perturb_buf(&self, out: &mut CudaSlice<u16>, m: usize) -> Result<()> {
        let call = MOE_ULP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let n = m * self.cfg.hidden_size;
        self.dsync()?;
        let mut v: Vec<u16> = self.dtoh(&*out)?;
        for (j, b) in v[..n].iter_mut().enumerate() {
            let mut z = (j as u64) ^ (call << 24);
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let e = (*b >> 10) & 0x1F;
            if z & 1 == 1 && e != 0 && e != 0x1F { *b ^= 1; }
        }
        self.dev.htod_sync_copy_into(&v, out)?;
        Ok(())
    }
}
