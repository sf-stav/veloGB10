//! DFlash2 / DFlash draft-side probes: synthetic draft generation, the draft probes with their mirror
//! reference, the round loaders (`load_df2_round*`), the S3T3 prompt set and the spec-scheduler builder.

use super::*;

/// S2F golden dump: one case = a directory with `manifest.json` + raw little-endian blobs
/// (written by `tool_probe/dflash2_golden.py` from the vendor reference).
struct GoldenCase {
    name: String,
    ctx_len: usize,
    anchor: u32,
    dtype: String,
    th: Vec<f32>,            // [C, hidden]
    layer_outs: Vec<Vec<f32>>, // 5 × [block, hidden]
    h_final: Vec<f32>,       // [block, hidden]
    logits: Vec<f32>,        // [7, vocab]
    unary: Vec<f32>,         // [7, 16]
    candidates: Vec<u32>,    // [7, 16]
    path: Vec<u32>,          // [7]
}

fn read_golden_case(dir: &std::path::Path) -> Result<GoldenCase, anyhow::Error> {
    let manifest = std::fs::read_to_string(dir.join("manifest.json"))
        .map_err(|e| anyhow::anyhow!("read {}: {e}", dir.join("manifest.json").display()))?;
    let m: serde_json::Value = serde_json::from_str(&manifest)?;
    let get = |name: &str| -> Result<(String, Vec<usize>), anyhow::Error> {
        let t = &m["tensors"][name];
        anyhow::ensure!(!t.is_null(), "manifest missing tensor {name}");
        Ok((
            t["file"].as_str().ok_or_else(|| anyhow::anyhow!("{name}.file"))?.to_string(),
            t["shape"].as_array().ok_or_else(|| anyhow::anyhow!("{name}.shape"))?
                .iter().map(|v| v.as_u64().unwrap() as usize).collect(),
        ))
    };
    let read_f32 = |name: &str| -> Result<Vec<f32>, anyhow::Error> {
        let (file, shape) = get(name)?;
        let raw = std::fs::read(dir.join(&file))?;
        let n: usize = shape.iter().product();
        anyhow::ensure!(raw.len() == n * 4, "{name}: {} bytes != {}*4", raw.len(), n);
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let read_u32 = |name: &str| -> Result<Vec<u32>, anyhow::Error> {
        let (file, shape) = get(name)?;
        let raw = std::fs::read(dir.join(&file))?;
        let n: usize = shape.iter().product();
        anyhow::ensure!(raw.len() == n * 4, "{name}: {} bytes != {}*4", raw.len(), n);
        Ok(raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let mut layer_outs = Vec::new();
    for l in 0..5 {
        layer_outs.push(read_f32(&format!("layer_out_{l}"))?);
    }
    Ok(GoldenCase {
        name: m["case"].as_str().unwrap_or("?").to_string(),
        ctx_len: m["ctx_len"].as_u64().ok_or_else(|| anyhow::anyhow!("ctx_len"))? as usize,
        anchor: m["anchor"].as_u64().ok_or_else(|| anyhow::anyhow!("anchor"))? as u32,
        dtype: m["dtype"].as_str().unwrap_or("f32").to_string(),
        th: read_f32("th")?,
        layer_outs,
        h_final: read_f32("h_final")?,
        logits: read_f32("logits")?,
        unary: read_f32("unary")?,
        candidates: read_u32("candidates")?,
        path: read_u32("path")?,
    })
}

/// S2F — `--gen-dflash2-synth <dir>`: the deterministic synthetic 81-tensor artifact generator.
pub(crate) fn run_gen_dflash2_synth(args: &[String], dir: &str) {
    let seed: u64 = parse_arg(args, "--seed").and_then(|s| s.parse().ok())
        .unwrap_or_else(gb10_inference::dflash2::synth::default_seed);
    let s = gb10_inference::dflash2::synth::generate(dir, seed).expect("generate synth artifact");
    println!("DFlash2 synthetic artifact generated (deterministic, seed {seed}):");
    println!("  dir       = {}", s.dir);
    println!("  sha256    = {}", s.sha256);
    println!("  tensors   = {} (assert 81)", s.n_tensors);
    println!("  params    = {} (asserted vs the live geometry's reconcile)", s.n_params);
    println!("  file_size = {} bytes", s.file_size);
    println!("  data_size = {} bytes (BF16 x params)", s.n_params * 2);
    println!("  header    = {} bytes (8-byte len + JSON)", s.header_size);
    println!("  config.json + SYNTHETIC_README.md written");
    println!("RESULT: GENERATED ({} tensors / {} params asserted at generation time)",
             gb10_inference::dflash2::n_tensors(), gb10_inference::dflash2::n_params());
}

/// S2F — `--probe-dflash2 <dir>`: load the REAL (or synthetic) artifact and run the oracle
/// checks. Prints PASS/FAIL per check; exit 0 only on all-PASS.
pub(crate) fn run_probe_dflash2(args: &[String], dir: &str) {
    use gb10_inference::dflash2::oracle::{Dflash2Config, Dflash2Oracle, RoundCtx};
    use gb10_inference::dflash2::synth::SyntheticTables;

    let mut all_pass = true;
    let mut check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass = false; }
    };

    // Load + validate (inventory/shapes/dtypes + sha256 pin; default pin = the REAL artifact).
    let sha_pin = match parse_arg(args, "--sha256") {
        Some(s) if s == "off" => None,
        Some(s) => Some(s),
        None => Some(gb10_inference::dflash2::REAL_SHA256),
    };
    let art = gb10_inference::dflash2::load::load(dir, sha_pin).expect("load artifact");
    println!("inventory: {} tensors, {} params, {} bytes, sha256 {}",
             art.n_tensors, art.n_params, art.file_size, art.sha256);
    check(&format!("inventory == {} tensors / {} params / BF16-only (loader, live geometry)",
                   gb10_inference::dflash2::n_tensors(), gb10_inference::dflash2::n_params()),
          art.n_tensors == gb10_inference::dflash2::n_tensors()
              && art.n_params == gb10_inference::dflash2::n_params());
    check(&format!("sha256 == a published pin (matched {}..{} — or --sha256 override)",
                   &art.sha256[..8], &art.sha256[56..]),
          sha_pin.is_none() || gb10_inference::dflash2::KNOWN_SHA256.iter().any(|k| *k == art.sha256.as_str())
              || sha_pin.map(|p| art.sha256.eq_ignore_ascii_case(p)).unwrap_or(false));

    let cfg = Dflash2Config::default();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), art.weights.clone()).expect("oracle");

    // Deterministic synthetic context (DECISION O): C committed positions of tap hiddens.
    let tap_dim = cfg.n_taps * cfg.hidden;
    let taps_gen = SyntheticTables::new(gb10_inference::dflash2::SYNTH_TAP_SEED);
    let tap_scale = 1.0f32 / (tap_dim as f32).sqrt();
    let gen_taps = |c: usize| -> Vec<f32> {
        let mut t = Vec::with_capacity(c * tap_dim);
        for i in 0..c {
            t.extend_from_slice(&taps_gen.row(SyntheticTables::TABLE_TAPS, i as u32, tap_dim, tap_scale));
        }
        t
    };
    let anchor: u32 = 12345;

    // ---- small case: C = 37 (SWA window inactive) -----------------------------
    let c_small = 37usize;
    let taps_s = gen_taps(c_small);
    let ctx_s = RoundCtx { tap_hiddens: taps_s.clone(), anchor };

    // (a) determinism — two run_rounds bit-identical.
    let a1 = oracle.run_round(&ctx_s);
    let a2 = oracle.run_round(&ctx_s);
    let det_ok = a1.th == a2.th && a1.layer_hiddens == a2.layer_hiddens && a1.h == a2.h
        && a1.logits == a2.logits && a1.select.tokens == a2.select.tokens
        && a1.select.candidates == a2.select.candidates && a1.select.unary == a2.select.unary
        && a1.select.scores == a2.select.scores;
    check("determinism: two run_rounds bit-identical", det_ok);

    // (b) wiring — sign-flip one layer-0 q_proj weight → outputs MUST change (anti-empty-compare;
    // the S2 R4 lesson: a 1-ulp perturb rounds away through the norms — a sign flip propagates).
    let mut pw = art.weights.clone();
    let q0s = pw.layers[0].q_proj.as_mut_slice();
    let idx = q0s.iter().position(|&x| x != 0.0).expect("a nonzero q_proj weight exists");
    q0s[idx] = -q0s[idx];
    let oracle_p = Dflash2Oracle::from_weights(cfg.clone(), pw).expect("perturbed oracle");
    let b = oracle_p.run_round(&ctx_s);
    let wiring_ok = b.logits != a1.logits || b.h != a1.h || b.select.tokens != a1.select.tokens;
    check("wiring: sign-flip one q_proj weight changes outputs", wiring_ok);

    // (c) incremental == batch tap projection (DECISION B), row-at-a-time.
    let th_batch = oracle.tap_project(&taps_s, c_small);
    let mut th_inc = Vec::with_capacity(c_small * cfg.hidden);
    for i in 0..c_small {
        let mut t = oracle.tap_project(&taps_s[i * tap_dim..(i + 1) * tap_dim], 1);
        th_inc.append(&mut t);
    }
    check("incremental == batch tap projection (bit-identical)", th_batch == th_inc && th_batch == a1.th);

    // (d) incremental == batch draft KV write (DECISION B), chunks 17 + 20 vs one-shot.
    let kv_batch = oracle.draft_kv_write(&th_batch, c_small, 0);
    let mut kv_inc = oracle.draft_kv_write(&th_batch[..17 * cfg.hidden], 17, 0);
    kv_inc.append(oracle.draft_kv_write(&th_batch[17 * cfg.hidden..], c_small - 17, 17));
    let kv_ok = kv_batch.layers.iter().zip(kv_inc.layers.iter())
        .all(|(x, y)| x.k == y.k && x.v == y.v && x.m == y.m);
    check("incremental == batch draft KV write, chunks 17+20 (bit-identical)", kv_ok);

    // (e) SWA-2048 window boundary (DECISION C), C = 2100: ctx row 52 is outside every block
    // query's band (q_pos−k_pos < 2048 ⇔ j ≥ 53+i) → perturbing it changes NOTHING (bit-exact);
    // row 53 is visible to block query 0 → perturbing it MUST change the output.
    let c_big = 2100usize;
    let taps_b = gen_taps(c_big);
    let big_base = oracle.run_round(&RoundCtx { tap_hiddens: taps_b.clone(), anchor });
    let mut taps_52 = taps_b.clone();
    taps_52[52 * tap_dim] = 1.2345678;
    let big_52 = oracle.run_round(&RoundCtx { tap_hiddens: taps_52, anchor });
    let masked_ok = big_52.h == big_base.h && big_52.logits == big_base.logits
        && big_52.select.tokens == big_base.select.tokens;
    check("window: perturbing MASKED ctx row 52 (C=2100) is bit-identical", masked_ok);
    let mut taps_53 = taps_b.clone();
    taps_53[53 * tap_dim] = 1.2345678;
    let big_53 = oracle.run_round(&RoundCtx { tap_hiddens: taps_53, anchor });
    let visible_ok = big_53.h != big_base.h || big_53.logits != big_base.logits;
    check("window: perturbing VISIBLE ctx row 53 (C=2100) changes outputs", visible_ok);

    // (f) structure.
    let logits_finite = a1.logits.iter().all(|x| x.is_finite());
    let hiddens_finite = a1.h.iter().all(|x| x.is_finite())
        && a1.layer_hiddens.iter().all(|lh| lh.iter().all(|x| x.is_finite()))
        && a1.th.iter().all(|x| x.is_finite());
    let tokens_in_vocab = a1.select.tokens.iter().all(|&t| (t as usize) < cfg.vocab);
    check("structure: logits + hiddens finite", logits_finite && hiddens_finite);
    check("structure: draft tokens < vocab", tokens_in_vocab);
    // candidates are exactly the top-16 of each logits row under the (value desc, id asc) total
    // order — verified via an INDEPENDENT full sort (the oracle used the incremental selector).
    let mut cand_ok = true;
    for p in 0..cfg.block - 1 {
        let row = &a1.logits[p * cfg.vocab..(p + 1) * cfg.vocab];
        let mut order: Vec<(f32, u32)> = row.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
        order.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        for kk in 0..16 {
            if order[kk].1 != a1.select.candidates[p][kk]
                || order[kk].0 != a1.select.unary[p][kk]
            {
                cand_ok = false;
            }
        }
    }
    check("structure: candidates == independent top-16 (value desc, id asc) + unary matches", cand_ok);
    // conv mechanics (DECISION M): prepare actually transforms its input; finish its output.
    let l0 = &art.weights.layers[0];
    let hn = {
        // input_layernorm of the block embeddings (piecewise re-derivation, C=37 case).
        let hidden = cfg.hidden;
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut emb = Vec::with_capacity(cfg.block * hidden);
        emb.extend_from_slice(&oracle.synth().row(SyntheticTables::TABLE_EMBED, anchor, hidden, scale));
        for _ in 1..cfg.block {
            emb.extend_from_slice(&oracle.synth().row(SyntheticTables::TABLE_EMBED, cfg.mask_token_id, hidden, scale));
        }
        let _ = &emb;
        // call the public pieces directly
        let prep = oracle.conv_prepare(&l0.attention_conv, &emb, cfg.block);
        let fin = oracle.conv_finish(&l0.attention_conv, &prep.x_conv, &prep.dyn_hold, cfg.block);
        (emb, prep, fin)
    };
    let conv_ok = hn.1.x_conv != hn.0 && hn.1.x_conv.iter().all(|x| x.is_finite())
        && hn.2 != hn.1.x_conv && hn.2.iter().all(|x| x.is_finite());
    check("structure: conv prepare/finish transform their inputs (wired, finite)", conv_ok);
    // verify width follows the LIVE block (DECISION G — the S5F fold note: whatever the
    // artifact, its inventory ships no confidence head; 27B 81 tensors, Ornith 96).
    let no_conf = gb10_inference::dflash2::inventory()
        .iter().all(|(n, _)| !n.contains("confidence"));
    check("structure: verify width ≡ live block (no confidence head in the inventory)",
          no_conf);

    // (g) piecewise — each API callable standalone, and the composition == run_round bit-exactly.
    let piecewise = (|| -> Result<(), String> {
        let hidden = cfg.hidden;
        let th = oracle.tap_project(&taps_s, c_small);
        let kv = oracle.draft_kv_write(&th, c_small, 0);
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut emb = Vec::with_capacity(cfg.block * hidden);
        emb.extend_from_slice(&oracle.synth().row(SyntheticTables::TABLE_EMBED, anchor, hidden, scale));
        for _ in 1..cfg.block {
            emb.extend_from_slice(&oracle.synth().row(SyntheticTables::TABLE_EMBED, cfg.mask_token_id, hidden, scale));
        }
        // layer-level piecewise: layer_forward(0) == layer_hiddens[0]
        let block_pos: Vec<usize> = (c_small..c_small + cfg.block).collect();
        let h1 = oracle.layer_forward(0, &emb, &kv.layers[0], &block_pos);
        if h1 != a1.layer_hiddens[0] { return Err("layer_forward(0) != run_round layer 0".into()); }
        let (layer_hiddens, h) = oracle.backbone_forward(&emb, &kv, c_small);
        if layer_hiddens != a1.layer_hiddens || h != a1.h {
            return Err("backbone_forward != run_round".into());
        }
        let h_sel = &h[hidden..cfg.block * hidden];
        let logits = oracle.logits(h_sel, cfg.block - 1);
        if logits != a1.logits { return Err("logits != run_round".into()); }
        let sel = oracle.select_path(h_sel, &logits, anchor);
        if sel.tokens != a1.select.tokens || sel.candidates != a1.select.candidates
            || sel.unary != a1.select.unary || sel.scores != a1.select.scores
        {
            return Err("select_path != run_round".into());
        }
        Ok(())
    })();
    check("piecewise: all pieces callable + composition == run_round", piecewise.is_ok());
    if let Err(e) = &piecewise { println!("           ({e})"); }

    // (h) golden — `--golden <dir>`: diff against the vendor-reference dump (per-case).
    if let Some(gdir) = parse_arg(args, "--golden") {
        let gpath = std::path::Path::new(gdir);
        let mut cases: Vec<std::path::PathBuf> = std::fs::read_dir(gpath)
            .expect("read golden dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir() && p.join("manifest.json").exists())
            .collect();
        cases.sort();
        if cases.is_empty() && gpath.join("manifest.json").exists() {
            cases.push(gpath.to_path_buf());
        }
        println!("golden: {} case(s) under {gdir}", cases.len());
        for case_dir in cases {
            let g = read_golden_case(&case_dir).expect("read golden case");
            let c = g.ctx_len;
            println!("  case {} (C={c}, anchor={}, dtype={}):", g.name, g.anchor, g.dtype);
            assert_eq!(g.anchor, anchor, "golden anchor must match the probe's");
            let taps = gen_taps(c);
            let out = oracle.run_round(&RoundCtx { tap_hiddens: taps, anchor });
            let f32_case = g.dtype == "f32";
            // continuous surfaces
            let r_th = rel_l2(&out.th, &g.th);
            let mut r_layers = Vec::new();
            for l in 0..5 {
                r_layers.push(rel_l2(&out.layer_hiddens[l], &g.layer_outs[l]));
            }
            let r_h = rel_l2(&out.h, &g.h_final);
            let r_logits = rel_l2(&out.logits, &g.logits);
            let r_max = r_layers.iter().copied().fold(r_th.max(r_h).max(r_logits), f64::max);
            println!("    rel-L2: th {:.3e} | layers [{}] | h {:.3e} | logits {:.3e}",
                     r_th, r_layers.iter().map(|r| format!("{:.3e}", r)).collect::<Vec<_>>().join(", "),
                     r_h, r_logits);
            // f32 cases are the FIDELITY gate (the oracle is the f32 definition). bf16 cases
            // DOCUMENT the reference's own dtype rounding (report-only — gating f32-vs-bf16
            // would conflate the reference's bf16 noise with port fidelity; measured gap is
            // ~1e-1 at the logits, i.e. the reference's own f32→bf16 sensitivity, not an
            // oracle error — the f32 case gates at 1e-3 and achieves ~4e-5).
            if f32_case {
                check(&format!("golden {}: continuous rel-L2 ≤ 1e-3 (max {r_max:.3e})", g.name),
                      r_max <= 1e-3);
            } else {
                println!("    (bf16 case: report-only — reference dtype gap, not gated)");
            }
            // argmax agreement per position
            let mut argmax_match = 0;
            for p in 0..7 {
                let mine = gb10_inference::dflash2::oracle::argmax(
                    &out.logits[p * cfg.vocab..(p + 1) * cfg.vocab]);
                let theirs = gb10_inference::dflash2::oracle::argmax(
                    &g.logits[p * cfg.vocab..(p + 1) * cfg.vocab]);
                if mine == theirs { argmax_match += 1; }
            }
            println!("    logits argmax match: {argmax_match}/7");
            // candidate SETS (sorted by id) — exact for f32, informational for bf16
            let mut sets_ok = true;
            for p in 0..7 {
                let mut mine: Vec<u32> = out.select.candidates[p].to_vec();
                let mut theirs: Vec<u32> = g.candidates[p * 16..(p + 1) * 16].to_vec();
                mine.sort_unstable();
                theirs.sort_unstable();
                if mine != theirs { sets_ok = false; }
            }
            println!("    candidate sets equal: {sets_ok}");
            // chain mechanics: feed the REFERENCE's (candidates, unary) through our chain
            let mut cand_arr = [[0u32; 16]; gb10_inference::dflash2::oracle::MAX_SELECT_LEVELS];
            let mut unary_arr = [[0.0f32; 16]; gb10_inference::dflash2::oracle::MAX_SELECT_LEVELS];
            for p in 0..(cfg.block - 1) {
                cand_arr[p].copy_from_slice(&g.candidates[p * 16..(p + 1) * 16]);
                unary_arr[p].copy_from_slice(&g.unary[p * 16..(p + 1) * 16]);
            }
            let h_sel = &out.h[cfg.hidden..cfg.block * cfg.hidden];
            let chained = oracle.select_chain(h_sel, &cand_arr, &unary_arr, anchor);
            let chain_path_ok = chained.tokens_v() == &g.path[..];
            println!("    chain-from-reference-candidates path exact: {chain_path_ok}");
            let own_path: Vec<u32> = out.select.tokens_v().to_vec();
            println!("    own-order path == reference path: {}", own_path == g.path);
            if f32_case {
                check(&format!("golden {}: logits argmax 7/7", g.name), argmax_match == 7);
                check(&format!("golden {}: candidate sets exact", g.name), sets_ok);
                check(&format!("golden {}: chain path exact (reference candidates)", g.name),
                      chain_path_ok);
                check(&format!("golden {}: own-order path == reference path", g.name),
                      own_path == g.path);
            }
        }
    }

    println!("RESULT: {}", if all_pass { "ALL PASS" } else { "FAIL" });
    if !all_pass { std::process::exit(1); }
}

/// The bf16-staged mirror reference for one round (every surface the GPU pass emits).
struct MirrorRef {
    th: Vec<f32>,
    layer_hiddens: Vec<Vec<f32>>,
    h: Vec<f32>,
    layer0: gb10_inference::dflash2::mirror::MirrorLayerOut,
    k0: Vec<f32>,
    v0: Vec<f32>,
}

fn mirror_reference(
    cfg: &gb10_inference::dflash2::oracle::Dflash2Config,
    w: &gb10_inference::dflash2::oracle::Dflash2Weights,
    synth: &gb10_inference::dflash2::synth::SyntheticTables,
    taps: &[f32],
    anchor: u32,
    c: usize,
) -> MirrorRef {
    use gb10_inference::dflash2::mirror as m;
    let inv = m::inv_freq(cfg);
    let (cos, sin) = m::rope_tables_half(cfg, &inv, c + gb10_inference::dflash2::block());
    let taps_bf16 = m::rb_clone(taps);
    let (_, th) = m::tap_project_mirror(cfg, &w.fc, &w.hidden_norm, &taps_bf16, c);
    let block_pos: Vec<usize> = (c..c + gb10_inference::dflash2::block()).collect();
    let emb = m::block_emb_mirror(cfg, synth, anchor);
    let mut ctx = Vec::new();
    for l in &w.layers {
        let (k, v) = m::draft_kv_mirror(cfg, l, &th, c, 0, &cos, &sin);
        ctx.push((k, v));
    }
    let mut h = emb.clone();
    let mut layer_hiddens = Vec::new();
    let mut layer0: Option<m::MirrorLayerOut> = None;
    for (li, l) in w.layers.iter().enumerate() {
        let out = m::mirror_layer_forward(cfg, l, &h, &ctx[li].0, &ctx[li].1, &block_pos, &cos, &sin);
        h = out.h3.clone();
        layer_hiddens.push(out.h3.clone());
        if li == 0 {
            layer0 = Some(out);
        }
    }
    let h_final = m::rb_clone(&m::rms_norm_rows(&h, &w.norm, gb10_inference::dflash2::block(), cfg.hidden, cfg.rms_eps));
    MirrorRef {
        th,
        layer_hiddens,
        h: h_final,
        layer0: layer0.unwrap(),
        k0: ctx[0].0.clone(),
        v0: ctx[0].1.clone(),
    }
}

/// S3F — `--probe-df2-draft <dir>`: run the DFlash2 draft-block forward ON GPU and diff it
/// against the bf16-staged mirror of the oracle (per-piece + whole-pass), then negative controls,
/// determinism and a perf pass.
pub(crate) fn run_probe_df2_draft(args: &[String], dir: &str) {
    use gb10_inference::dflash2::gpu::Df2Gpu;
    use gb10_inference::dflash2::oracle::{Dflash2Config, Dflash2Oracle};
    use gb10_inference::dflash2::synth::SyntheticTables;

    let all_pass = std::cell::Cell::new(true);
    let check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass.set(false); }
    };

    let art = gb10_inference::dflash2::load::load(dir, Some(gb10_inference::dflash2::REAL_SHA256))
        .expect("load artifact");
    println!("inventory: {} tensors, {} params, sha256 {}",
             art.n_tensors, art.n_params, &art.sha256[..16]);
    check(&format!("inventory == {} tensors / {} params (live geometry)",
                   gb10_inference::dflash2::n_tensors(), gb10_inference::dflash2::n_params()),
          art.n_tensors == gb10_inference::dflash2::n_tensors()
              && art.n_params == gb10_inference::dflash2::n_params());

    let cfg = Dflash2Config::default();
    let _oracle = Dflash2Oracle::from_weights(cfg.clone(), art.weights.clone()).expect("oracle");
    let synth = SyntheticTables::new(gb10_inference::dflash2::SYNTH_EMBED_HEAD_SEED);

    let tap_dim = cfg.n_taps * cfg.hidden;
    let taps_gen = SyntheticTables::new(gb10_inference::dflash2::SYNTH_TAP_SEED);
    let tap_scale = 1.0f32 / (tap_dim as f32).sqrt();
    let gen_taps = |c: usize| -> Vec<f32> {
        let mut t = Vec::with_capacity(c * tap_dim);
        for i in 0..c {
            t.extend_from_slice(&taps_gen.row(SyntheticTables::TABLE_TAPS, i as u32, tap_dim, tap_scale));
        }
        t
    };
    let anchor: u32 = 12345;

    let max_c = 4096usize;
    let mut gpu = Df2Gpu::load(dir, max_c).expect("gpu load");

    let cs = [37usize, 512, 2100, 4096];

    println!("\n== per-piece + whole-pass diffs (device vs bf16-staged mirror) ==");
    for &c in &cs {
        let taps = gen_taps(c);
        let r = mirror_reference(&cfg, &art.weights, &synth, &taps, anchor, c);
        let dev = gpu.forward(&taps, anchor, c, 2048, true).expect("gpu forward");
        let p = dev.pieces.as_ref().expect("pieces");

        println!("\n-- C={c} --");
        let per_piece = |name: &str, dev: &[f32], mir: &[f32], gate: f64| {
            let rl = rel_l2(dev, mir);
            println!("    {name}: rel-L2 {rl:.3e} [{}<={gate}]", if rl <= gate { "PASS" } else { "FAIL" });
            if rl > gate { all_pass.set(false); }
        };
        per_piece("q_proj", &p.q, &r.layer0.q, 1e-3);
        per_piece("k_proj", &p.k, &r.layer0.k, 1e-3);
        per_piece("v_proj", &p.v, &r.layer0.v, 1e-3);
        per_piece("kernel_proj (dyn)", &p.dyn_attn, &r.layer0.dyn_attn, 1e-3);
        per_piece("o_proj", &p.o, &r.layer0.o, 1e-3);
        per_piece("conv prepare", &p.x_conv, &r.layer0.x_conv, 1e-3);
        per_piece("conv finish", &p.fin, &r.layer0.fin, 1e-3);
        per_piece("attention", &p.attn, &r.layer0.attn, 1e-3);
        per_piece("mlp down", &p.mlp_out, &r.layer0.down, 1e-3);
        per_piece("input_ln", &p.input_ln_out, &r.layer0.input_ln_out, 1e-3);
        per_piece("post_ln", &p.post_ln_out, &r.layer0.post_ln_out, 1e-3);
        per_piece("mlp conv prepare", &p.x_conv2, &r.layer0.x_conv2, 1e-3);
        per_piece("mlp conv finish", &p.fin2, &r.layer0.fin2, 1e-3);
        per_piece("k_ctx", &p.k_ctx, &r.k0, 1e-3);
        per_piece("v_ctx", &p.v_ctx, &r.v0, 1e-3);

        let r_th = rel_l2(&dev.th, &r.th);
        println!("    th: rel-L2 {r_th:.3e} [{}<=5e-3]", if r_th <= 5e-3 { "PASS" } else { "FAIL" });
        if r_th > 5e-3 { all_pass.set(false); }
        for li in 0..5 {
            let rl = rel_l2(&dev.layer_hiddens[li], &r.layer_hiddens[li]);
            println!("    layer {li} hidden: rel-L2 {rl:.3e} [{}<=5e-3]", if rl <= 5e-3 { "PASS" } else { "FAIL" });
            if rl > 5e-3 { all_pass.set(false); }
        }
        let r_h = rel_l2(&dev.h, &r.h);
        println!("    final h: rel-L2 {r_h:.3e} [{}<=1e-2]", if r_h <= 1e-2 { "PASS" } else { "FAIL" });
        if r_h > 1e-2 { all_pass.set(false); }
    }

    println!("\n== negative controls (must FIRE) ==");
    // (a) sign-flip one uploaded layer-0 q_proj weight -> diff must explode (C=37).
    {
        let c = 37;
        let taps = gen_taps(c);
        let r = mirror_reference(&cfg, &art.weights, &synth, &taps, anchor, c);
        let base = gpu.forward(&taps, anchor, c, 2048, true).expect("base forward");
        let mut flipped = art.weights.clone();
        let q0 = flipped.layers[0].q_proj.as_mut_slice();
        let idx = q0.iter().position(|&x| x != 0.0).expect("nonzero q_proj");
        q0[idx] = -q0[idx];
        gpu.set_layer0_q_proj(&flipped.layers[0].q_proj);
        let pert = gpu.forward(&taps, anchor, c, 2048, true).expect("pert forward");
        // restore
        gpu.set_layer0_q_proj(&art.weights.layers[0].q_proj);
        let d_q = rel_l2(&pert.pieces.as_ref().unwrap().q, &base.pieces.as_ref().unwrap().q);
        let d_h = rel_l2(&pert.h, &base.h);
        let base_ok = rel_l2(&base.pieces.as_ref().unwrap().q, &r.layer0.q) <= 1e-3;
        println!("    sign-flip q_proj: device-vs-device q rel-L2 {d_q:.3e}, h {d_h:.3e} [must be non-zero]; base-vs-mirror still {base_ok}");
        check("negative control: sign-flip weight fires (q/h diff non-zero)", d_q > 1e-6 && d_h > 1e-6 && base_ok);
    }
    // (b) drop the band lower bound at C=4096 -> attention must explode (the window-boundary lesson).
    {
        let c = 4096;
        let taps = gen_taps(c);
        let r = mirror_reference(&cfg, &art.weights, &synth, &taps, anchor, c);
        let banded = gpu.forward(&taps, anchor, c, 2048, true).expect("banded");
        let noband = gpu.forward(&taps, anchor, c, c + 8 + 1000, true).expect("no-band");
        let d_attn = rel_l2(&banded.pieces.as_ref().unwrap().attn, &noband.pieces.as_ref().unwrap().attn);
        let banded_ok = rel_l2(&banded.pieces.as_ref().unwrap().attn, &r.layer0.attn) <= 1e-3;
        println!("    drop band lower bound (C=4096): band-vs-noband attn rel-L2 {d_attn:.3e} [must be >> 1e-3]; banded-vs-mirror still {banded_ok}");
        check("negative control: dropping the band fires (attn diff explodes)", d_attn > 1e-2 && banded_ok);
    }

    println!("\n== determinism (two passes bit-identical) ==");
    {
        let c = 37;
        let taps = gen_taps(c);
        let a1 = gpu.forward(&taps, anchor, c, 2048, false).expect("d1");
        let a2 = gpu.forward(&taps, anchor, c, 2048, false).expect("d2");
        let det = a1.th == a2.th && a1.layer_hiddens == a2.layer_hiddens && a1.h == a2.h;
        if !det {
            if a1.th != a2.th {
                let i = a1.th.iter().zip(&a2.th).position(|(x, y)| x != y).unwrap_or(0);
                println!("    th differs at {i}: {} vs {}", a1.th[i], a2.th[i]);
            }
            for li in 0..5 {
                if a1.layer_hiddens[li] != a2.layer_hiddens[li] {
                    let i = a1.layer_hiddens[li].iter().zip(&a2.layer_hiddens[li]).position(|(x, y)| x != y).unwrap_or(0);
                    println!("    layer {li} differs at {i}: {} vs {}", a1.layer_hiddens[li][i], a2.layer_hiddens[li][i]);
                }
            }
            if a1.h != a2.h {
                let i = a1.h.iter().zip(&a2.h).position(|(x, y)| x != y).unwrap_or(0);
                println!("    final h differs at {i}: {} vs {}", a1.h[i], a2.h[i]);
            }
        }
        check("two passes bit-identical (th + per-layer + final h)", det);
    }

    println!("\n== perf (whole pass, C=4096, median/p10/p90) ==");
    {
        let c = 4096;
        let taps = gen_taps(c);
        // warmup
        for _ in 0..10 {
            gpu.forward(&taps, anchor, c, 2048, false).expect("warmup");
        }
        let mut ms = Vec::with_capacity(100);
        for _ in 0..100 {
            let t0 = std::time::Instant::now();
            gpu.forward(&taps, anchor, c, 2048, false).expect("perf");
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| ms[(f * (ms.len() - 1) as f64) as usize];
        println!("    whole pass C=4096 (100 reps): median {:.3} ms  p10 {:.3}  p90 {:.3}",
                 q(0.5), q(0.10), q(0.90));
        println!("    (Step-0 skinny GEMM stream = 15.21 ms/round @ 236 GB/s; fc alone = 1.12 ms @ R=4)");
    }

    println!("\nRESULT: {}", if all_pass.get() { "ALL PASS" } else { "FAIL" });
    if !all_pass.get() { std::process::exit(1); }
}

// ===========================================================================================
// ===========================================================================================
// S5F — `load_df2_round`: build the DFlash2 engine-integration set (the round + the tap-sink
// twins) from the trunk, arming the trunk's verify-time capture. Returns None on ANY failure —
// the caller serves via MTP (the standing-directive fallback; never a hard failure).
// ===========================================================================================
pub(super) fn load_df2_round(gpu: &mut gb10_inference::gpu::GpuModel, max_c: usize)
    -> Option<(gb10_inference::dflash2::round::Df2Round,
               std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>,
               std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>)> {
    // The head (and the single-node paths) resolve --draft-dir from the CLI; the TP node uses
    // the shipped config's dir (load_df2_round_dir) — never its own argv.
    let args = std::env::args().collect::<Vec<_>>();
    let draft_dir = required_dir_arg(&args, "--draft-dir",
                                     "the DFlash2 draft artifact (spec-source is a DFlash2 source)");
    // --sha256 <hex|off> overrides the published artifact pin (retrained selectors carry a new
    // hash; "off" disables the sha check — the inventory/shape/dtype guard still runs).
    let sha_pin = parse_arg(&args, "--sha256");
    load_df2_round_dir(gpu, max_c, &draft_dir, sha_pin)
}

/// S9F (the TP-DF2 leg): `load_df2_round` with an EXPLICIT draft dir — the node's path (the
/// head's resolved dir ships on the config; the artifact bytes must be identical on every rank
/// or the round drafts diverge and the verify all-reduces desync). `sha_pin` = the artifact
/// sha256 pin override: None = published REAL_SHA256, Some("off") = no sha check, Some(hex) =
/// pin to that hash (the `--sha256` flag; rides TpConfig so head and node agree).
pub(super) fn load_df2_round_dir(gpu: &mut gb10_inference::gpu::GpuModel, max_c: usize, draft_dir: &str,
                      sha_pin: Option<&str>)
    -> Option<(gb10_inference::dflash2::round::Df2Round,
               std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>,
               std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>)> {
    use gb10_inference::dflash2::capture::{Df2PrimeSink, Df2TapSink};
    use gb10_inference::dflash2::round::Df2Round;
    // DFLASH2-GEN: install the draft's geometry from its config.json BEFORE the guards — the
    // ctx bound, the trunk-compat check and the load below all read the LIVE geometry, and a
    // non-27B draft would otherwise be refused against the 27B defaults (the 2026-10-23 boot
    // receipt: the Ornith trunk was WARNed against hidden=5120 exact=64). Idempotent: the
    // round load installs the same values again.
    let _ = gb10_inference::dflash2::geom::install_geom_from_dir(draft_dir);
    // S10R — the ctx guard: the S10R window-bounding fix made the round's attention smem
    // ctx-INDEPENDENT (band_smem(), 8,864 B constant — PLAN/B8_S10R_DISSECTION.md §4), so the
    // round now serves up to the model's max_position_embeddings. MAX_CTX_SAFE = 262,144 is the
    // RoPE-table envelope: above it the round's positions would exceed the model's trained
    // range — AUTO-FALL BACK to MTP (the standing directive's fallback, never a hard cap).
    // Same inputs on every rank (max_seq_len rides TpConfig), so the head and the zero-config
    // nodes fall back in lockstep.
    if max_c > gb10_inference::dflash2::max_ctx_safe() {
        eprintln!("[df2] WARN: --max-seq-len {max_c} exceeds the DFlash2 round's ctx bound \
                   ({} = the model's max_position_embeddings; the round's RoPE tables end there) — \
                   serving via MTP (auto-fallback)",
                  gb10_inference::dflash2::max_ctx_safe());
        return None;
    }
    // S10' §4 — the trunk-compatibility guard: the round is dimension-fixed to the 3.8-27B shapes
    // (HIDDEN 5120 / VOCAB 248320 / tap layers 5..61). An incompatible trunk (any older model)
    // would OOB the borrowed head/embed at the first round — refuse to load and serve via MTP
    // (the same standing-directive fallback as an absent/failed artifact).
    if !gpu.df2_round_compatible() {
        let (h, v, l) = gpu.df2_trunk_dims();
        eprintln!("[df2] WARN: trunk hidden={h} vocab={v} layers={l} is not DFlash2-compatible \
                   (the round needs hidden={} vocab>={} layers>{} exact={}) — serving via MTP \
                   (auto-fallback; the round's geometry comes from the draft config.json)",
                   gb10_inference::dflash2::hidden(), gb10_inference::dflash2::vocab(),
                   gb10_inference::dflash2::geom().tap_layers.last().copied().unwrap_or(0),
                   gb10_inference::dflash2::num_target_layers());
        return None;
    }
    // S9F (the TP-DF2 leg): the round now loads on EVERY rank under TP — the trunk's full
    // pre-shard lm_head is kept for the borrow ([df2-tp]=1, gpu.rs tp_shard_weights), the
    // drafter artifact ships via --draft-dir / the shipped config, and every rank drafts the
    // SAME tokens (bit-identical taps from the all-reduced hiddens) so the verify all-reduces
    // stay in lockstep. The residency decision rides the post-load CalibTable message.
    let (head, embed) = match gpu.df2_borrow() {
        Some(p) => p,
        None => {
            eprintln!("[df2] WARN: trunk lm_head/embed are not NVFP4/BF16 (or absent) — the DFlash2 \
                       borrowed-head path is unavailable; serving via MTP");
            return None;
        }
    };
    // P2 (round sharding): `--df2-round-shard on` rides TpConfig (SPMD — the head ships it,
    // the zero-config node resolves the identical flag from the shipped config). The AR
    // context exists only at world > 2 post-attach (gpu.df2_ar_ctx); at world <= 2 or
    // single-box the flag is a NO-OP with a log line (the quad is the campaign target).
    // A sharded-load failure is a load failure — the standing MTP fallback applies on every
    // rank in lockstep (the CalibTable df2_round outcome ships the head's result).
    let ar = {
        let want = gb10_inference::tp::tp_config().map(|c| c.df2_round_shard).unwrap_or(false);
        let ctx = gpu.df2_ar_ctx();
        match (want, ctx) {
            (true, Some(a)) => {
                println!("[df2] round-shard: ENGAGED (world={}, rank={}) — sharded drafter load", a.world, a.rank);
                Some(a)
            }
            (true, None) => {
                println!("[df2] round-shard: requested but NOT engaged (no TP attach or world <= 2) \
                          — loading the REPLICATED round (flag is a no-op here)");
                None
            }
            (false, _) => None,
        }
    };
    let mut round = match gb10_inference::dflash2::round::Df2Round::load_tp_pinned(
        draft_dir, Some(head), Some(embed), max_c, ar, sha_pin) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[df2] WARN: DFlash2 round load FAILED ({e:#}) — absent/failed artifact is \
                       never a hard failure; serving via MTP (standing directive)");
            return None;
        }
    };
    let sink = std::sync::Arc::new(Df2TapSink::new(gpu.dev()));
    round.attach_sink(&sink);
    let prime = std::sync::Arc::new(Df2PrimeSink::new(
        gpu.dev(), gb10_inference::batch::PREFILL_CHUNK));
    gpu.set_df2_capture(sink.clone());
    Some((round, sink, prime))
}

// ===========================================================================================
// S5F — the on-engine harnesses: greedy bit-identity (--probe-df2-lossless), the DFlash2-source
// chi-square gate (--bench-df2-sample), and the τ matrix (--bench-df2-matrix).
// ===========================================================================================

/// The S3T3 domain prompt sets, VERBATIM from `tool_probe/dflash2_domain.py::PROMPTS` — the
/// committed matrix template (S3T3 R6). (prompt_id, domain, text).
pub(super) const S3T3_PROMPTS: &[(&str, &str, &str)] = &[
    // GSM8K-style few-shot math
    ("gsm1", "math", "Question: A farmer has 12 chickens. Each chicken lays 2 eggs per day. \
How many eggs does the farmer collect in 5 days?\n\
reasoning: Each chicken lays 2 eggs per day, so 12 chickens lay 12 * 2 = 24 eggs per day. \
Over 5 days, the farmer collects 24 * 5 = 120 eggs.\n\
#### 120\n\n\
Question: A store sells notebooks for 3 dollars each and pens for 2 dollars each. \
Sara buys 4 notebooks and 6 pens. How much does she spend in total?\n"),
    ("gsm2", "math", "Question: A train travels at 80 km/h for 3 hours and then at 60 km/h for 2 hours. \
How far does it travel in total?\n\
reasoning: In the first part the train covers 80 * 3 = 240 km. \
In the second part it covers 60 * 2 = 120 km. \
The total distance is 240 + 120 = 360 km.\n\
#### 360\n\n\
Question: A concert ticket costs 45 dollars. A group buys 3 tickets and also pays a \
15 dollar booking fee. How much does the group pay in total?\n"),
    ("gsm3", "math", "Question: A jacket costs 80 dollars. The price is reduced by 25%. What is the new price?\n\
reasoning: 25% of 80 dollars is 80 * 0.25 = 20 dollars. \
The new price is 80 - 20 = 60 dollars.\n\
#### 60\n\n\
Question: A phone costs 600 dollars. During a sale the price is reduced by 15%. \
How much does the phone cost during the sale?\n"),
    ("gsm4", "math", "Question: Tom is 3 times as old as his son. The son is 9 years old. How old is Tom?\n\
reasoning: Tom's age is 3 * 9 = 27 years.\n\
#### 27\n\n\
Question: A recipe for 4 people needs 300 grams of flour. \
How many grams of flour are needed for 10 people?\n"),
    ("gsm5", "math", "Question: A baker packs 48 cookies into boxes of 6. How many boxes does she fill?\n\
reasoning: 48 cookies divided by 6 per box gives 48 / 6 = 8 boxes.\n\
#### 8\n\n\
Question: 117 students are going on a trip. Each bus holds 45 students. \
How many buses are needed?\n"),
    // HumanEval/MBPP-style code completion
    ("cod1", "code", "def find_most_frequent(nums):\n\
    \"\"\"Return the element that appears most often in the list nums.\n\
    If several elements tie, return the smallest one.\"\"\"\n"),
    ("cod2", "code", "def is_balanced(text):\n\
    \"\"\"Return True if the string text has balanced parentheses: every '(' is closed\n\
    by a matching ')' and no ')' appears before its matching '('.\"\"\"\n"),
    ("cod3", "code", "def unique_sorted(nums):\n\
    \"\"\"Return a sorted list of the distinct elements of nums, in ascending order.\"\"\"\n"),
    ("cod4", "code", "def sum_of_squares(n):\n\
    \"\"\"Return the sum 1^2 + 2^2 + ... + n^2 for a positive integer n.\"\"\"\n"),
    ("cod5", "code", "def count_vowels(text):\n\
    \"\"\"Return the number of vowels (a, e, i, o, u) in the string text, ignoring case.\"\"\"\n"),
    // S3T/S3T2 chat controls (the ~1.9 anchor)
    ("chat1", "chat", "Explain how a transformer's self-attention mechanism works, in three short sentences."),
    ("chat2", "chat", "Describe a quiet morning in a mountain village, in a few sentences."),
    ("chat3", "chat", "What is the difference between a stack and a queue, and when would you use each?"),
    // S5F4 — the prose/essay cell (S6F input; the MiaAI quality flag: "DSpark gives the essay
    // back; DFlash2 does not"). Babbage→GPUs-class long-form essay continuation: the model
    // continues mid-essay (prose coherence is the eyeball metric, τ the acceptance metric).
    ("ess1", "prose", "It is a striking fact of the history of computing that the analytical engine \
of Charles Babbage, conceived in the 1830s, already contained the logical architecture of the \
modern computer: a store for data, a mill for arithmetic, and a program controlled by punched \
cards. What Babbage lacked was not vision but materials and the patience of his contemporaries. \
A century later, when the electronic computer finally emerged from the wartime laboratories, it \
did so not as a single invention but as a convergence of mathematics, physics, and engineering. \
The general-purpose graphics processing unit that now powers deep learning descends from this \
lineage in a way that would have surprised even Babbage, who imagined his engines humming away \
at tables of logarithms rather than at the weights of neural networks. To understand how the GPU \
came to be the workhorse of artificial intelligence, one must trace three threads: the economics \
of video game graphics, the collapse of Dennard scaling, and the improbable reuse of SIMD \
machinery for matrix multiplication. Each of these threads, taken separately, is a story of \
pragmatic engineering; taken together, they explain why the most important computing device of \
the twenty-first century was designed to draw pixels, and why its inventors had no idea what it \
would one day be used for."),
];

/// Build the S5F scheduler (one lane, forced+pinned policy, optional DFlash2 round) for the
/// on-engine harnesses. `gpu` is consumed; returns the scheduler + the round's ring ranges for
/// the no-alias assert (None when no round).
pub(super) fn build_spec_scheduler(gpu: gb10_inference::gpu::GpuModel, kv_stride: usize,
                        eos: Vec<u32>, spec_source: gb10_inference::batch::SpecSource,
                        mtp_depth: usize, df2: Option<gb10_inference::dflash2::round::Df2Round>,
                        sink: Option<std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>>,
                        prime: Option<std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>>,
                        dspark: Option<gb10_inference::dspark::round::DsparkRound>,
                        dspark_sink: Option<std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>>,
                        dspark_prime: Option<std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>>,
                        step_dump: Option<gb10_inference::dflash2::stepdump::StepDump>)
    -> gb10_inference::batch::BatchScheduler {
    use gb10_inference::batch::{BatchScheduler, MtpPolicy};
    let policy = MtpPolicy::with_source(gpu.mtp_present(), Some(true), Some(mtp_depth), vec![], spec_source);
    let (stx, srx) = tokio::sync::mpsc::unbounded_channel();
    std::mem::drop(stx);   // the bench drives admit directly
    BatchScheduler::with_df2(gpu, 1, kv_stride, eos, srx, policy, false, 0, false, false,
                             df2, sink, prime, dspark, dspark_sink, dspark_prime, step_dump)
}

/// `--probe-dflash <ctx-file> [tokens.json]`: E29-B1 DFlash drafter probe. Loads the
/// Hy3-DFlash-B8 draft model, runs the 8-token block forward per recorded ctx feature (the target's
/// post-layer hiddens at layers {1,20,39,58,77}), prints per-position top-1 + acceptance vs the
/// target chain, and dumps the logits (f32 LE) to `<ctx-file>.logits.bin` for the torch-golden
/// comparison. The LM head is the checkpoint's embed_tokens stand-in (the checkpoint has none).
pub(crate) fn run_probe_dflash(args: &[String]) {
    use gb10_inference::dflash as df;
    let ctx_file = parse_arg(args, "--probe-dflash")
        .expect("--probe-dflash <ctx-file> [tokens.json] requires a ctx-features file");
    let idx = args.iter().position(|a| a == "--probe-dflash").unwrap();
    let tokens_json = args.get(idx + 2)
        .map(|s| s.as_str())
        .filter(|s| !s.starts_with("--"));
    let model_dir = parse_arg(args, "--model-dir")
        .expect("--probe-dflash requires --model-dir <DIR>");
    let dcfg = df::DflashCfg::load(std::path::Path::new(model_dir)).expect("read drafter config.json");
    let input = df::read_probe_input(std::path::Path::new(ctx_file), tokens_json.map(std::path::Path::new), &dcfg)
        .expect("read ctx-features file");
    let max_pos = input.plen + input.steps.len() + dcfg.block + 16;
    let mut d = df::DflashDrafter::load_from_dir(std::path::Path::new(model_dir), max_pos)
        .expect("dflash load");
    eprintln!("[dflash] drafter: {} layers, h={}, heads {}:{} hd {}, inter {}, vocab {}, rms_eps {}, rope_theta {}, block {} (config-driven)",
              d.layers.len(), d.h, d.nh, d.nkv, d.hd, d.inter, d.vocab, d.rms_eps, d.rope_theta, d.block);
    eprintln!("[dflash] ctx file {}: plen={}, {} step(s)", ctx_file, input.plen, input.steps.len());
    let mut pool = gb10_inference::gpu::Pool::new(d.dev.clone());
    let out_path = format!("{}.logits.bin", ctx_file);
    let mut dump: Vec<f32> = Vec::new();
    for (i, step) in input.steps.iter().enumerate() {
        let mut kv = df::DflashKv::new(&d, step.ctx_len + d.block);
        let ctx_bf: Vec<half::bf16> = step.ctx.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let ctx_dev = d.dev.htod_sync_copy(&ctx_bf).expect("upload ctx feature");
        let t0 = std::time::Instant::now();
        let logits = d.forward(&mut pool, &mut kv, &ctx_dev, step.ctx_len,
                               &step.block_tokens, step.pos_start,
                               None::<&cudarc::driver::CudaSlice<half::bf16>>,
                               None::<&cudarc::driver::CudaSlice<half::bf16>>)
            .expect("dflash forward");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let top1 = d.top1(&logits);
        dump.extend_from_slice(&logits);
        println!("[dflash] step {i}: pos_start={} ctx_len={} block={} forward {ms:.2} ms",
                 step.pos_start, step.ctx_len, d.block);
        println!("[dflash] step {i} top-1: {}", top1.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
        if let Some(chain) = &step.chain {
            let acc: Vec<bool> = (0..d.block)
                .map(|k| chain.get(step.pos_start + k).map(|&t| t == top1[k]).unwrap_or(false))
                .collect();
            println!("[dflash] step {i} accept (top-1 == chain[plen+i+k]): {}",
                     acc.iter().map(|a| if *a { "y" } else { "n" }).collect::<Vec<_>>().join(" "));
        }
    }
    // Dump header "DFLG" + version u32 + nsteps u32 + vocab u32, then nsteps*BLOCK*vocab f32
    // (row-major per step, then per block position). Consumed by the torch-golden comparison.
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out_path).expect("create logits dump"));
    use std::io::Write;
    file.write_all(b"DFLG").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(input.steps.len() as u32).to_le_bytes()).unwrap();
    file.write_all(&(d.vocab as u32).to_le_bytes()).unwrap();
    for x in &dump { file.write_all(&x.to_le_bytes()).unwrap(); }
    file.flush().unwrap();
    eprintln!("[dflash] wrote {} logits to {}", dump.len(), out_path);
}
