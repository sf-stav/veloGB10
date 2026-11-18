//! `--probe-exl3-xtp`: the TP=1 reference writer (`probe_xtp_ref`) and the TP rank-identity run (`run_tp_xtp`)
//! with its comparison / sp-check helpers. See `mod.rs` for the map.

use super::*;

/// Metrics of a pass against the reference: per-step logits relL2 (median, max), per-layer
/// residual relL2 at cmp (median, max over "layer{li}"), first step whose argmax != ref token.
/// Router top-k set flips at cmp between two captures: (layer, #ids that differ, the REFERENCE's
/// logit gap between its k-th and (k+1)-th expert — a near-tie selection flips on tiny noise).
fn router_flips(ref_layers: &[(String, Vec<f32>)], layers: &[(String, Vec<f32>)], ne: usize, k: usize) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    for (n, rv) in ref_layers {
        let Some(li) = n.strip_prefix("rlog").and_then(|x| x.parse::<usize>().ok()) else { continue };
        let Some((_, cv)) = layers.iter().find(|(m, _)| m == n) else { continue };
        if rv.len() < ne || cv.len() < ne { continue; }
        let a = topk_ids(&rv[..ne], k + 1);
        let b = topk_ids(&cv[..ne], k);
        let sa: std::collections::BTreeSet<usize> = a[..k].iter().copied().collect();
        let sb: std::collections::BTreeSet<usize> = b.iter().copied().collect();
        let d = sa.difference(&sb).count();
        if d > 0 { out.push((li, d, rv[a[k - 1]] - rv[a[k]])); }
    }
    out
}

fn compare(ref_lg: &[Vec<f32>], ref_layers: &[(String, Vec<f32>)], ref_seq: &[u32], plen: usize,
           lg: &[Vec<f32>], layers: &[(String, Vec<f32>)], am: &[u32], t0: usize) -> (f64, f64, f64, f64, Option<usize>, Vec<(String, f64)>) {
    let mut lr: Vec<f64> = ref_lg.iter().zip(lg).map(|(r, c)| rel_l2(c, r)).collect();
    let lmax = lr.iter().cloned().fold(0f64, f64::max);
    let lmed = median(&mut lr);
    let mut per: Vec<(String, f64)> = Vec::new();
    for (n, rv) in ref_layers {
        if let Some((_, cv)) = layers.iter().find(|(m, _)| m == n) { per.push((n.clone(), rel_l2(cv, rv))); }
    }
    let mut hr: Vec<f64> = per.iter().filter(|(n, _)| n.starts_with("layer")).map(|(_, v)| *v).collect();
    let hmax = hr.iter().cloned().fold(0f64, f64::max);
    let hmed = median(&mut hr);
    let div = (plen - 1..t0 + am.len()).find(|&t| am[t - t0] != ref_seq[t + 1]);
    (lmed, lmax, hmed, hmax, div, per)
}

fn write_f32s(path: &str, rows: &[Vec<f32>]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("create {path}"))?);
    for r in rows { for v in r { f.write_all(&v.to_le_bytes())?; } }
    Ok(())
}

fn read_f32s(path: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("read {path}"))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct XtpRef {
    prompt_len: usize,
    gen: usize,
    seq: Vec<u32>,
    vocab: usize,
    cmp: usize,
    layer_names: Vec<String>,
    layer_lens: Vec<usize>,
    max_pos: usize,
    floor_bit: u32,
    floor_logits_med: f64,
    floor_logits_max: f64,
    floor_hidden_med: f64,
    floor_hidden_max: f64,
    floor_div: Option<usize>,
    rerun_bitexact: bool,
    ident: String,
    text: String,
    /// the reduction-site floor (one f16 ulp on every MoE output, ~half the elements)
    #[serde(default)]
    moeulp_logits_med: f64,
    #[serde(default)]
    moeulp_logits_max: f64,
    #[serde(default)]
    moeulp_hidden_med: f64,
    #[serde(default)]
    moeulp_hidden_max: f64,
    #[serde(default)]
    moeulp_div: Option<usize>,
    #[serde(default)]
    floor_per_layer: Vec<(String, f64)>,
    #[serde(default)]
    moeulp_per_layer: Vec<(String, f64)>,
    /// the floor ENSEMBLE (kind, seed, logits med, hidden med, first router flip (layer, ids, gap))
    #[serde(default)]
    band: Vec<(String, u64, f64, f64, Option<(usize, usize, f32)>)>,
    /// TP-B: the program options the reference ran with (the TP=2 run must match pf_chunk)
    #[serde(default)]
    opts_mode: String,
    /// TP-B: first traced decode step (plen-1 when the prompt went through the chunked prefill)
    #[serde(default)]
    t0: usize,
    /// TP-B: untraced timing (TP=1): prefill s, eager ms/token, graphed ms/token (NOT perf claims)
    #[serde(default)]
    timing: Option<(f64, f64, f64)>,
    /// TP-B: the TP=1 gates report (G-T1-c/d control at TP=1)
    #[serde(default)]
    gates: Vec<String>,
}

/// Host top-k of router logits (ties -> lower id), the load-balance analysis of the deal.
fn topk_ids(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// `--probe-exl3-xtp` (single box, TP=1): the reference trace + the Q1-style floor, written to
/// `out_dir` for the TP=2 run to compare against. Also reports the static-deal load balance
/// (busiest-rank share of the top-k per (layer, token)) for both deals on the SAME trace.
/// TP-B: `opts.pf_chunk` = the prompt goes through the served chunked prefill (every pass and
/// every floor member); `opts.gates` = the TP=1 control of G-T1-c/d; `opts.time_tokens` = an
/// untraced timing pass (eager + graphed ms/token, prefill s).
pub fn probe_xtp_ref(dir: &str, prompt: &[u32], gen: usize, max_pos: usize, out_dir: &str, opts: &XtpOpts) -> Result<()> {
    crate::dprintln!("{}", identity_line("xtp-ref"));
    crate::dprintln!("XTP_REF program {} (prompt {} tokens, gen {gen})", opts.to_mode(), prompt.len());
    let model = FwdModel::load(dir, 1, max_pos)?;
    let mut sc = FwdModel::scratch(&model.dev, &model.cfg, 1, model.max_pos())?;
    let mut psc_store = match opts.pf_chunk { Some(c) => Some(model.prefill_scratch(c)?), None => None };
    let plen = prompt.len();
    anyhow::ensure!(plen >= 2 && gen >= 1, "xtp: need a prompt of >= 2 tokens and gen >= 1");
    let cmp = plen - 1;
    let ne = model.cfg.num_experts;
    let topk = model.cfg.num_experts_per_tok;
    // deal balance accumulators: [interleave, contig] x (sum of busiest share, count)
    let mut bal = [(0f64, 0usize); 2];
    let t0 = std::time::Instant::now();
    let p = trace_pass(&model, &mut sc, psc_store.as_mut(), false, prompt, None, gen, cmp, None, true, &mut |_, _| Ok(()),
        &mut |_t, ent, _am| {
            for (n, v) in ent {
                if n.starts_with("rlog") && v.len() >= ne {
                    let ids = topk_ids(&v[..ne], topk);
                    for (di, kind) in ["interleave", "contig"].iter().enumerate() {
                        let mut c = [0usize; 2];
                        for &e in &ids { c[ep_owner(e, ne, 2, kind)] += 1; }
                        bal[di].0 += c[0].max(c[1]) as f64 / topk as f64;
                        bal[di].1 += 1;
                    }
                }
            }
            Ok(())
        })?;
    let (seq, am, lg, layers, tt0) = (p.seq, p.am, p.lg, p.at_cmp, p.t0);
    crate::dprintln!("XTP_REF pass: {} steps from t0 {tt0} in {:.1}s (eager, captures on; chunked prefill {:.2}s)",
             am.len(), t0.elapsed().as_secs_f32(), p.pf_s);
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{dir}/tokenizer.json"))?;
    let text = tok.decode(&seq[plen..], false).unwrap_or_default();
    crate::dprintln!("XTP_REF greedy[{gen}]: {:?}", text);
    for (di, kind) in ["interleave", "contig"].iter().enumerate() {
        crate::dprintln!("XTP_DEAL {kind}: mean busiest-rank share of the top-{topk} per (layer, token) = {:.4} over {} routings (ideal 0.5)",
                 bal[di].0 / bal[di].1.max(1) as f64, bal[di].1);
    }
    // determinism control: a plain teacher-forced rerun must be bit-exact
    let p2 = trace_pass(&model, &mut sc, psc_store.as_mut(), false, prompt, Some(&seq), gen, cmp, None, true,
                        &mut |_, _| Ok(()), &mut |_, _, _| Ok(()))?;
    let rerun_bitexact = p2.am == am && p2.lg.iter().zip(&lg).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    crate::dprintln!("XTP_REF rerun (teacher-forced, no perturbation): {}", if rerun_bitexact { "BIT-EXACT" } else { "NOT bit-exact (nondeterministic reference!)" });
    // The floor ENSEMBLE (Q1-style, measured in THIS run): for each seed, (a) bit-0 = one f32 ulp
    // flipped in ~half of every GDN S element after every step (Q1's floor), (b) moe-ulp = one
    // f16 ulp flipped in ~half of every MoE output (the site TP=2 reassociates). A single seed's
    // layer-median is dominated by WHERE the first near-tie router flip lands (a discrete event),
    // so the band is measured over seeds, not assumed.
    let seeds: u64 = crate::opts::var(crate::opt!("xtp-floor-seeds")).ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let floor_bit = 0u32;
    let mut band: Vec<(String, u64, f64, f64, Option<(usize, usize, f32)>)> = Vec::new();
    let mut seed0: [(f64, f64, f64, f64, Option<usize>, Vec<(String, f64)>); 2] = Default::default();
    let lay_only = |v: &Vec<(String, f64)>| v.iter().filter(|(n, _)| n.starts_with("layer")).cloned().collect::<Vec<_>>();
    for seed in 0..seeds {
        for (ki, kind) in ["bit0", "moe-ulp"].iter().enumerate() {
            let res = if ki == 0 {
                trace_pass(&model, &mut sc, psc_store.as_mut(), false, prompt, Some(&seq), gen, cmp, Some((floor_bit, seed)), true,
                           &mut |_, _| Ok(()), &mut |_, _, _| Ok(()))
            } else {
                MOE_ULP.store((seed << 32) | 1, std::sync::atomic::Ordering::Relaxed);
                let r = trace_pass(&model, &mut sc, psc_store.as_mut(), false, prompt, Some(&seq), gen, cmp, None, true,
                                   &mut |_, _| Ok(()), &mut |_, _, _| Ok(()));
                MOE_ULP.store(0, std::sync::atomic::Ordering::Relaxed);
                r
            };
            let pf = res?;
            let c = compare(&lg, &layers, &seq, plen, &pf.lg, &pf.at_cmp, &pf.am, pf.t0);
            let flips = router_flips(&layers, &pf.at_cmp, ne, topk);
            crate::dprintln!("XTP_FLOOR {kind} seed {seed}: logits relL2 med {:.3e} max {:.3e} | hidden relL2 med {:.3e} max {:.3e} | argmax div {:?} | first router flip {:?} ({} flipped layers)",
                     c.0, c.1, c.2, c.3, c.4.map(|d| d as i64 - (plen as i64 - 1)), flips.first(), flips.len());
            if seed == 0 {
                for chunk in lay_only(&c.5).chunks(12) {
                    crate::dprintln!("XTP_FLOOR {kind} seed 0 per-layer hidden: {}", chunk.iter().map(|(n, v)| format!("{n}={v:.2e}")).collect::<Vec<_>>().join(" "));
                }
                seed0[ki] = c.clone();
            }
            band.push((kind.to_string(), seed, c.0, c.2, flips.first().copied()));
        }
    }
    let (flm, flx, fhm, fhx, fdiv, fper) = seed0[0].clone();
    let (mlm, mlx, mhm, mhx, mdiv, mper) = seed0[1].clone();
    let mut bl: Vec<f64> = band.iter().map(|b| b.2).collect();
    let bh_max = band.iter().map(|b| b.3).fold(0f64, f64::max);
    let bh_min = band.iter().map(|b| b.3).fold(f64::INFINITY, f64::min);
    crate::dprintln!("XTP_BAND {} members: logits relL2 med-of-meds {:.3e} | hidden relL2 med range [{bh_min:.3e}, {bh_max:.3e}] | first flip layers {:?}",
             band.len(), median(&mut bl), band.iter().map(|b| b.4.map(|f| f.0)).collect::<Vec<_>>());
    // TP-B: the untraced timing pass (eager AND graphed at TP=1; TP=2 runs eager) — not a perf claim
    let timing = if opts.time_tokens > 1 {
        crate::opts::set(crate::opt!("exl3-no-graph"), "1");
        let (pf_e, ms_e, ids_e) = time_pass(&model, &mut sc, psc_store.as_mut(), prompt, opts.time_tokens, &mut |_, _| Ok(()))?;
        crate::opts::unset(crate::opt!("exl3-no-graph"));
        let (_, ms_g, ids_g) = time_pass(&model, &mut sc, psc_store.as_mut(), prompt, opts.time_tokens, &mut |_, _| Ok(()))?;
        crate::dprintln!("XTP_TIME TP=1: prefill {pf_e:.3}s ({} tok{}) | decode eager {ms_e:.2} ms/token, graphed {ms_g:.2} ms/token over {} tokens | eager==graphed ids: {}",
                 plen, if opts.pf_chunk.is_some() { ", chunked" } else { ", token-by-token" }, opts.time_tokens - 1, ids_e == ids_g);
        Some((pf_e, ms_e, ms_g))
    } else { None };
    // TP-B: the TP=1 control of G-T1-c / G-T1-d (same function the TP=2 run executes)
    let gates = if opts.gates > 0 {
        let (ok, rep) = gates_bv_state(&model, psc_store.as_mut(), prompt, opts.gates, &mut |_, v| Ok(v))?;
        for l in &rep { crate::dprintln!("TP1 {l}"); }
        crate::dprintln!("TP1 GATES {} (the TP=1 control)", if ok { "EXACT" } else { "DIVERGES" });
        rep
    } else { Vec::new() };
    std::fs::create_dir_all(out_dir)?;
    write_f32s(&format!("{out_dir}/xtp_ref_logits.f32"), &lg)?;
    write_f32s(&format!("{out_dir}/xtp_ref_layers.f32"), &layers.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>())?;
    let r = XtpRef {
        prompt_len: plen, gen, seq: seq.clone(), vocab: lg.first().map_or(0, |v| v.len()), cmp,
        layer_names: layers.iter().map(|(n, _)| n.clone()).collect(),
        layer_lens: layers.iter().map(|(_, v)| v.len()).collect(), max_pos,
        floor_bit, floor_logits_med: flm, floor_logits_max: flx, floor_hidden_med: fhm, floor_hidden_max: fhx,
        floor_div: fdiv, rerun_bitexact, ident: identity_line("xtp-ref"), text,
        moeulp_logits_med: mlm, moeulp_logits_max: mlx, moeulp_hidden_med: mhm, moeulp_hidden_max: mhx, moeulp_div: mdiv,
        floor_per_layer: lay_only(&fper), moeulp_per_layer: lay_only(&mper), band,
        opts_mode: opts.to_mode(), t0: tt0, timing, gates,
    };
    std::fs::write(format!("{out_dir}/xtp_ref.json"), serde_json::to_string_pretty(&r)?)?;
    crate::dprintln!("XTP_REF written to {out_dir} (seq {} tokens, {} logits rows, {} layer captures)", seq.len(), lg.len(), layers.len());
    Ok(())
}

/// Names of captured buffers that are NOT replicated across ranks (never part of G-T1-b):
/// yd (the rank's local experts' outputs) and ysh (its shared-expert slice when sharded).
fn rank_local_capture(name: &str) -> bool { name.starts_with("yd") || name.starts_with("ysh") }

/// TP-SP2: tallies of the redesigned G-T1-b over the sequence-parallel chunks.
#[derive(Default)]
struct SpTally { chunks: usize, r_n: usize, r_bad: usize, g_n: usize, g_bad: usize, o_n: usize, o_bad: usize, p_n: usize, p_bad: usize }

/// TP-SP2 — G-T1-b for a SEQUENCE-PARALLEL chunk (row ownership: the streams, the reduce-scattered
/// mixer / MoE outputs, the inject coefficients and the final mix hold only a rank's OWN rows, so they
/// are no longer replicated; the redundancy the replicated gate used is restored by re-running the
/// chunk replicated, prefill_prompt_chunks). `ent` = the replicated reference run's captures (`R:`
/// names: whole buffers + every rank block `<name>@<b>` of the own-row classes) + the sequence-
/// parallel run's (whole buffers + this rank's own block). One positional exchange of every digest
/// (both ranks build the same list: the same names, own blocks in the same class order), then:
///   (R) every R whole buffer (rank-local ones excluded) equal across ranks — the replicated gate;
///   (G) every S gathered / all-row buffer (not an own-row class) equal across ranks AND to this
///       rank's R buffer of the same name;
///   (O) every S own-row block equal to this rank's R block AND to the PEER's R block of those rows —
///       a divergence of one rank's owned rows (compute, reduce-scatter or gather) cannot hide;
///   (P) the peer's S own-row blocks equal to this rank's R blocks (both ranks reach one verdict).
/// S whole buffers of the own-row classes are not compared (their non-owned rows are stale by design).
fn sp_check(tag: &str, ent: &Caps, names_bad: &mut BTreeMap<String, usize>, first_bad: &mut Option<String>,
            t: &mut SpTally, rank: usize) -> Result<(usize, usize)> {
    let hash = |v: &[f32]| { let h = fnv64(v); [h as u32, (h >> 32) as u32] };
    let cls = |n: &str| n.split('@').next().unwrap_or(n).trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
    let r_whole: Vec<&(String, Vec<f32>)> = ent.iter()
        .filter(|(n, _)| n.starts_with("R:") && !n.contains('@') && !rank_local_capture(&n[2..])).collect();
    let r_blk: Vec<&(String, Vec<f32>)> = ent.iter().filter(|(n, _)| n.starts_with("R:") && n.contains('@')).collect();
    let s_glob: Vec<&(String, Vec<f32>)> = ent.iter()
        .filter(|(n, _)| !n.starts_with("R:") && !n.contains('@') && !rank_local_capture(n) && !sp_row_class(n)).collect();
    let s_own: Vec<&(String, Vec<f32>)> = ent.iter().filter(|(n, _)| !n.starts_with("R:") && n.contains('@')).collect();
    let mine: Vec<u32> = r_whole.iter().chain(&r_blk).chain(&s_glob).chain(&s_own).flat_map(|(_, v)| hash(v)).collect();
    // TP-4S: every rank's digest list (world 2: exactly the pairwise exchange; world 4: the hub all-gather)
    let all = crate::net::exchange_u32s_all(&mine, mine.len().max(4) + 4)?;
    anyhow::ensure!(rank < all.len(), "{tag}: rank {rank} outside the {}-rank digest exchange", all.len());
    for (p, v) in all.iter().enumerate() {
        anyhow::ensure!(v.len() == mine.len(), "{tag}: rank {p}'s TP-SP2 digest list has {} words, this rank's {}", v.len(), mine.len());
    }
    let at = |v: &[u32], i: usize| (v[2 * i], v[2 * i + 1]);
    let (o_rb, o_sg, o_so) = (r_whole.len(), r_whole.len() + r_blk.len(), r_whole.len() + r_blk.len() + s_glob.len());
    let r_by_name: BTreeMap<&str, usize> = r_whole.iter().enumerate().map(|(i, (n, _))| (&n[2..], i)).collect();
    let rb_by_name: BTreeMap<&str, usize> = r_blk.iter().enumerate().map(|(i, (n, _))| (&n[2..], o_rb + i)).collect();
    let mut bad = 0usize;
    let flag = |what: &str, n: &str, names_bad: &mut BTreeMap<String, usize>, first_bad: &mut Option<String>| {
        *names_bad.entry(format!("{what}:{}", cls(n))).or_default() += 1;
        if first_bad.is_none() { *first_bad = Some(format!("{tag} {what} buffer {n}")); }
    };
    // (R) the replicated reference, whole buffers, across ALL ranks
    for (i, (n, _)) in r_whole.iter().enumerate() {
        t.r_n += 1;
        if all.iter().any(|p| at(&mine, i) != at(p, i)) { t.r_bad += 1; bad += 1; flag("R", &n[2..], names_bad, first_bad); }
    }
    // (G) gathered / all-row buffers of the SP run: across ALL ranks and vs this rank's R
    for (j, (n, _)) in s_glob.iter().enumerate() {
        t.g_n += 1;
        let i = o_sg + j;
        let r = r_by_name.get(n.as_str()).map(|&k| at(&mine, k));
        if all.iter().any(|p| at(&mine, i) != at(p, i)) || r != Some(at(&mine, i)) { t.g_bad += 1; bad += 1; flag("S", n, names_bad, first_bad); }
    }
    // (O) own-row blocks vs this rank's R block and EVERY other rank's R block of the same rows; (P) every other
    // rank's own-row blocks (its S list: same class order, that rank's block index) vs this rank's R blocks.
    // World 2: the one peer, block 1 - rank: the pre-TP-4S check exactly.
    for (j, (n, _)) in s_own.iter().enumerate() {
        let i = o_so + j;
        t.o_n += 1;
        match rb_by_name.get(n.as_str()) {
            Some(&k) if at(&mine, i) == at(&mine, k) && all.iter().all(|p| at(&mine, i) == at(p, k)) => {}
            _ => { t.o_bad += 1; bad += 1; flag("S-own", n, names_bad, first_bad); }
        }
        let base = n.split_once('@').map_or(n.as_str(), |(b, _)| b);
        for (p, pv) in all.iter().enumerate().filter(|&(p, _)| p != rank) {
            t.p_n += 1;
            let pn = format!("{base}@{p}");
            match rb_by_name.get(pn.as_str()) {
                Some(&k) if at(pv, i) == at(&mine, k) => {}
                _ => { t.p_bad += 1; bad += 1; flag("S-peer-own", &pn, names_bad, first_bad); }
            }
        }
    }
    t.chunks += 1;
    Ok((r_whole.len() + s_glob.len() + (1 + all.len().saturating_sub(1)) * s_own.len(), bad))
}

/// S-B9-REL-INTEG A5: the `G-T1-b RANK_IDENTITY FAIL` line printed when pass 1 ENDS IN AN ERROR (the per-step
/// agreement tripwire, a dead link): the class table of everything compared up to that step, then the error.
pub(super) fn rank_identity_abort_line(n_dig: usize, steps_seen: usize, steps_ok: usize, pf_dig: usize, pf_chunks: usize, pf_bad: usize,
                            names_bad: &BTreeMap<String, usize>, first_bad: Option<&str>, why: &str) -> String {
    format!("G-T1-b RANK_IDENTITY FAIL — pass 1 ABORTED after {steps_seen} decode steps ({n_dig} replicated decode buffers \
             compared{}); steps fully identical {steps_ok}/{steps_seen}; mismatches by class {:?}{}; the error: {why}",
            if pf_chunks > 0 { format!(", + {pf_dig} whole-chunk prefill digests over {pf_chunks} chunks ({pf_bad} bad)") } else { String::new() },
            names_bad,
            first_bad.map(|s| format!("; first: {s}")).unwrap_or_default())
}

/// The TP=W xtp run (W = ctx.world), every rank (SPMD). `ref_dir` = Some on rank 0 unless `ident` — it reads the
/// reference and broadcasts the teacher-forced sequence; the node receives it over the link.
/// `head_prompt` = (prompt, gen) on rank 0 for an `ident` run (no reference).
pub fn run_tp_xtp(pack_dir: &str, ctx: crate::tp::TpContext, ref_dir: Option<&str>, head_prompt: Option<(Vec<u32>, usize)>,
                  max_pos: usize, opts: &XtpOpts) -> Result<()> {
    let mut ctx = ctx;
    ctx.sanity()?;
    ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
    let rref: Option<XtpRef> = match ref_dir {
        Some(d) if !opts.ident_only => Some(serde_json::from_str(&std::fs::read_to_string(format!("{d}/xtp_ref.json"))
            .with_context(|| format!("read {d}/xtp_ref.json — run --probe-exl3-xtp (TP=1) first"))?)?),
        _ => None,
    };
    if let Some(r) = &rref {
        let ro = XtpOpts::parse(if r.opts_mode.is_empty() { "xtp" } else { &r.opts_mode })?;
        anyhow::ensure!(ro.pf_chunk == opts.pf_chunk,
            "xtp: the reference ran pf_chunk {:?}, this run {:?} — the decode steps would not align", ro.pf_chunk, opts.pf_chunk);
    }
    let head_seq: Option<(Vec<u32>, usize, usize)> = match (&rref, &head_prompt) {
        (Some(r), _) => Some((r.seq.clone(), r.gen, r.prompt_len)),
        (None, Some((p, g))) => { let mut s = p.clone(); s.resize(p.len() + g, 0); Some((s, *g, p.len())) }
        (None, None) => None,
    };
    let (seq, gen, plen) = match &head_seq {
        Some((s, g, p)) => ctx.broadcast_prompt(Some((s, *g, *p)))?,
        None => ctx.broadcast_prompt(None)?,
    };
    anyhow::ensure!(plen >= 2 && seq.len() == plen + gen, "xtp: bad broadcast (plen {plen}, gen {gen}, seq {})", seq.len());
    if let Some(r) = &rref { anyhow::ensure!(r.max_pos == max_pos, "xtp: max_pos {max_pos} != reference {}", r.max_pos); }
    world_gt2_opts_refusals(ctx.world, opts)?;
    let (rank, world, link) = ctx.into_parts();
    // The launch thread pinned to a big core (gpu.rs / tp_serve rule: an unpinned launch thread
    // presents exactly like a protocol stall) — the proxy takes core 19.
    anyhow::ensure!(crate::net::pin_thread(9), "launch thread failed to pin to core 9 — TP refuses to run unpinned");
    crate::dprintln!("[exl3-tp] rank {rank}/{world}: program {} — loading {pack_dir} (EP deal '{}', max_pos {max_pos}) ...",
             opts.to_mode(), ep_deal_kind());
    let t_load = std::time::Instant::now();
    let model = FwdModel::load_tp(pack_dir, 1, max_pos, Some(TpAttach { rank, world, link }))?;
    crate::dprintln!("[exl3-tp] rank {rank}/{world}: loaded in {:.1}s — trunk per rank: {} q heads / {} kv heads, GDN {} k / {} v heads",
             t_load.elapsed().as_secs_f32(), model.tc.num_heads, model.tc.num_kv_heads, model.tc.lin_num_k_heads, model.tc.lin_num_v_heads);
    let mut sc = FwdModel::scratch(&model.dev, &model.cfg, 1, model.max_pos())?;
    let mut psc_store = match opts.pf_chunk { Some(c) => Some(model.prefill_scratch(c)?), None => None };
    let cmp = plen - 1;
    let prompt = seq[..plen].to_vec();

    // ---- pass 1: teacher-forced on the reference sequence (free-running for `ident`); G-T1-b
    // digests every replicated buffer of every decode step AND every prefill chunk ----
    let mut names_bad: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_bad: Option<String> = None;
    let mut n_dig = 0usize;
    let mut steps_ok = 0usize;
    let mut steps_seen = 0usize;
    let (mut pf_dig, mut pf_bad, mut pf_chunks) = (0usize, 0usize, 0usize);
    let mut check = |tag: &str, ent: &Caps, names_bad: &mut BTreeMap<String, usize>, first_bad: &mut Option<String>| -> Result<(usize, usize)> {
        let rep: Vec<&(String, Vec<f32>)> = ent.iter().filter(|(n, _)| !rank_local_capture(n)).collect();
        let mine: Vec<u32> = rep.iter().flat_map(|(_, v)| { let h = fnv64(v); [h as u32, (h >> 32) as u32] }).collect();
        let all = crate::net::exchange_u32s_all(&mine, mine.len().max(4) + 4)?;
        let mut bad = 0;
        for (i, (n, _)) in rep.iter().enumerate() {
            if all.iter().any(|peer| mine[2 * i] != peer[2 * i] || mine[2 * i + 1] != peer[2 * i + 1]) {
                let cls = n.trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
                *names_bad.entry(cls).or_default() += 1;
                bad += 1;
                if first_bad.is_none() { *first_bad = Some(format!("{tag} buffer {n}")); }
            }
        }
        Ok((rep.len(), bad))
    };
    let mut nb2: BTreeMap<String, usize> = BTreeMap::new();
    let mut fb2: Option<String> = None;
    // TP-SP2: the sequence-parallel chunks' redesigned identity check (sp_check) — its own tallies
    let mut spt = SpTally::default();
    if let Some((fl, fr)) = opts.sp_fault {
        anyhow::ensure!(seq_parallel_on() && fr < world as usize && fl < model.layers.len(),
                        "--xtp-sp-fault {fl}:{fr} needs --tp-seq-parallel on, a rank < {world} and a layer < {}", model.layers.len());
        SP_FAULT.store((fl + 1) | (fr << 16), AtOrd::Relaxed);
        crate::dprintln!("[exl3-tp] rank {rank}: TP-SP2 G-T1-b fault injection ARMED: rank {fr} perturbs one own stream row after layer {fl} \
                  in the first split chunk's sequence-parallel run (the gate must FAIL)");
    }
    let t1 = std::time::Instant::now();
    let forced = if opts.ident_only { None } else { Some(seq.as_slice()) };
    // (a `let` of its own: the closures below are temporaries of THIS statement, so the tallies they borrow are free
    // again for the abort report; in a `match` scrutinee they would live to the end of the match)
    let r1 = trace_pass(&model, &mut sc, psc_store.as_mut(), true, &prompt, forced, gen, cmp, None, rank == 0,
        &mut |k, ent| {
            let (n, bad) = if ent.iter().any(|(n, _)| n.starts_with("R:")) {
                sp_check(&format!("prefill chunk {k}"), ent, &mut nb2, &mut fb2, &mut spt, rank as usize)?
            } else {
                check(&format!("prefill chunk {k}"), ent, &mut nb2, &mut fb2)?
            };
            pf_dig += n; pf_bad += bad; pf_chunks += 1;
            Ok(())
        },
        &mut |t, ent, am| {
            let (n, bad) = check(&format!("step {t}"), ent, &mut names_bad, &mut first_bad)?;
            n_dig += n;
            steps_seen += 1;
            if bad == 0 { steps_ok += 1; }
            // the per-step agreement tripwire (argmax + step), AGENTS §2.10
            probe_agree(t as u64, am).map_err(|why| anyhow::anyhow!("TP agree() failed at step {t} (argmax {am}): {why} — \
                ranks diverged or the link aborted"))?;
            Ok(())
        });
    let p1 = match r1 {
        Ok(p) => p,
        Err(e) => {
            // S-B9-REL-INTEG A5: a divergence that flips an argmax fails the per-step tripwire above and used to return
            // through `?` BEFORE the class table was printed; the digests of the steps compared so far are the evidence.
            for (k, v) in &nb2 { *names_bad.entry(format!("prefill:{k}")).or_default() += v; }
            if first_bad.is_none() { first_bad = fb2.clone(); }
            crate::dprintln!("{}", rank_identity_abort_line(n_dig, steps_seen, steps_ok, pf_dig, pf_chunks, pf_bad, &names_bad,
                                                    first_bad.as_deref(), &format!("{e:#}")));
            return Err(e);
        }
    };
    let pass1_s = t1.elapsed().as_secs_f64();
    let (am1, lg1, lay1, pt0) = (p1.am, p1.lg, p1.at_cmp, p1.t0);
    let total_steps = am1.len();
    for (k, v) in nb2 { *names_bad.entry(format!("prefill:{k}")).or_default() += v; }
    if first_bad.is_none() { first_bad = fb2; }
    let ident_ok = names_bad.is_empty() && n_dig > 0;
    crate::dprintln!("G-T1-b RANK_IDENTITY {} — {n_dig} replicated decode buffers over {total_steps} steps x (embed, 48 x [xpre, yatt, rlog, ymlp, layer], precollapse, finalmix, logits){}; \
              steps fully identical {steps_ok}/{total_steps}; mismatches by class {:?}{}",
             if ident_ok { "OK" } else { "FAIL" },
             if pf_chunks > 0 { format!(" + {pf_dig} whole-chunk prefill digests over {pf_chunks} chunks ({pf_bad} bad) x (embed, 48 x [xpre, rmid, yatt, xpre_mlp, inj, ymlp, layer], precollapse, finalmix)") } else { String::new() },
             names_bad,
             first_bad.map(|s| format!("; first: {s}")).unwrap_or_default());
    if spt.chunks > 0 {
        crate::dprintln!("G-T1-b SP (row ownership, TP-SP2) — {} sequence-parallel chunk(s), each run twice on the same state: \
                  R = the replicated reference ({} whole-buffer digests compared across ranks, {} bad) | S = the \
                  sequence-parallel run: {} gathered / all-row digests (embed, xpre, xpre_mlp, streams, hcarry, logits) \
                  == the peer's AND this rank's R ({} bad); {} own-row block digests (rmid, yatt, inj, ymlp, layer, \
                  precollapse, finalmix @ own rows) == this rank's R block AND the peer's R block ({} bad); {} peer \
                  own-row block digests == this rank's R block ({} bad) -> {}",
                 spt.chunks, spt.r_n, spt.r_bad, spt.g_n, spt.g_bad, spt.o_n, spt.o_bad, spt.p_n, spt.p_bad,
                 if spt.r_bad + spt.g_bad + spt.o_bad + spt.p_bad == 0 { "OK" } else { "FAIL" });
    }

    // ---- pass 2: free-running greedy (both ranks agree every step) ----
    let t2 = std::time::Instant::now();
    let p2 = trace_pass(&model, &mut sc, psc_store.as_mut(), false, &prompt, None, gen, cmp, None, false, &mut |_, _| Ok(()),
        &mut |t, _ent, am| {
            probe_agree((1 << 20) + t as u64, am).map_err(|why| anyhow::anyhow!("TP agree() failed in the greedy pass at \
                step {t} (argmax {am}): {why}"))?;
            Ok(())
        })?;
    let pass2_s = t2.elapsed().as_secs_f64();
    let seq2 = p2.seq;

    // comm latency (if traced): K2_GO - K1_IN per epoch = publish -> peer partial released
    let lat_line = |what: &str| -> String {
        if let Some((g, _c, _cnt, _gw, dev_epoch)) = crate::net::trace_data() {
            let n = (dev_epoch as usize).min(crate::net::GTS_EPOCHS);
            let mut v: Vec<f64> = Vec::new();
            for e in (dev_epoch as usize + 1 - n)..=(dev_epoch as usize) {
                let s = (e % crate::net::GTS_EPOCHS) * crate::net::GTS_STRIDE;
                let (k1o, k2g, k1i) = (g[s + crate::net::GTS_K1_OUT], g[s + crate::net::GTS_K2_GO], g[s + crate::net::GTS_K1_IN]);
                if k1o > 0 && k2g >= k1i && k1i > 0 { v.push((k2g - k1i) as f64 / 1e3); }
            }
            if !v.is_empty() {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
                return format!("barrier latency {what} K1-in -> K2-go (GPU clock, last {} epochs of {dev_epoch}): p50 {:.1} us p90 {:.1} us p99 {:.1} us",
                               v.len(), p(0.5), p(0.9), p(0.99));
            }
        }
        "barrier latency: tracing off (--tp-trace=1 / --tp-trace to measure)".to_string()
    };
    crate::dprintln!("[exl3-tp] rank {rank}: {}", lat_line("(traced passes)"));
    crate::dprintln!("[exl3-tp] rank {rank}: pass1 {total_steps} steps from t0 {pt0} {pass1_s:.1}s (digest capture on; chunked prefill {:.2}s), pass2 greedy {pass2_s:.1}s (captures on, eager)",
             p1.pf_s);

    // ---- TP-B: untraced timing pass (eager; not a perf claim) ----
    if opts.time_tokens > 1 {
        let (pf_s, ms, ids) = time_pass(&model, &mut sc, psc_store.as_mut(), &prompt, opts.time_tokens, &mut |i, am| {
            probe_agree((2 << 20) + i as u64, am).map_err(|why| anyhow::anyhow!("TP agree() failed in the timing pass at \
                token {i} (argmax {am}): {why}"))?;
            Ok(())
        })?;
        crate::dprintln!("XTP_TIME TP={world} rank {rank}: prefill {pf_s:.3}s ({plen} tok{}) | decode eager {ms:.2} ms/token over {} tokens (48 x 3 all-reduces per token){}",
                 if opts.pf_chunk.is_some() { ", chunked" } else { ", token-by-token" }, opts.time_tokens - 1,
                 rref.as_ref().and_then(|r| r.timing).map(|(p1s, e, g)| format!(" | TP=1 ref: prefill {p1s:.3}s, eager {e:.2} ms/token, graphed {g:.2} ms/token")).unwrap_or_default());
        let tokz = crate::tokenizer::QwenTokenizer::from_file(&format!("{pack_dir}/tokenizer.json"))?;
        if rank == 0 { crate::dprintln!("XTP_TIME greedy text: {:?}", tokz.decode(&ids, false).unwrap_or_default()); }
        crate::dprintln!("[exl3-tp] rank {rank}: {}", lat_line("(incl. the untraced timing pass)"));
    }

    // ---- TP-B: G-T1-c / G-T1-d at TP=W ----
    let mut gates_ok = true;
    if opts.gates > 0 {
        let (ok, rep) = gates_bv_state(&model, psc_store.as_mut(), &prompt, opts.gates, &mut |tag, v| {
            let all = crate::net::exchange_u32s_all(&[tag as u32, v], 8)?;
            for (r, peer) in all.iter().enumerate() {
                anyhow::ensure!(peer[0] == tag as u32, "gates: rank {r} at check {:#x}, this rank at {tag:#x} — desync", peer[0]);
            }
            // The caller compares the returned word with its own: hand back a differing rank's word if any.
            Ok(all.iter().map(|p| p[1]).find(|&w| w != v).unwrap_or(v))
        })?;
        for l in &rep { crate::dprintln!("TP{world} rank {rank} {l}"); }
        let peer_ok = crate::net::exchange_u32s_all(&[ok as u32], 4)?
            .iter().enumerate().filter(|(r, _)| *r as i32 != rank).all(|(_, p)| p[0] != 0);
        gates_ok = ok && peer_ok;
        crate::dprintln!("TP{world} GATES rank {rank}: this rank {} | {} => {}", if ok { "EXACT" } else { "DIVERGES" },
                 if world == 2 { format!("peer {}", if peer_ok { "EXACT" } else { "DIVERGES" }) }
                 else { format!("all peers {}", if peer_ok { "EXACT" } else { "DIVERGE" }) },
                 if gates_ok { format!("G-T1-c/d EXACT at TP={world}") } else { "G-T1-c/d FAIL".to_string() });
        if let Some(r) = &rref { for l in &r.gates { crate::dprintln!("  (TP=1 control) {l}"); } }
    }

    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{pack_dir}/tokenizer.json"))?;
    if let (Some(r), Some(d)) = (&rref, ref_dir) {
        // ---- the cross-TP tolerance tier vs the TP=1 reference ----
        let rl = read_f32s(&format!("{d}/xtp_ref_logits.f32"))?;
        let ref_lg: Vec<Vec<f32>> = rl.chunks_exact(r.vocab).map(|c| c.to_vec()).collect();
        let rlay = read_f32s(&format!("{d}/xtp_ref_layers.f32"))?;
        let mut ref_layers = Vec::new();
        let mut off = 0;
        for (n, &len) in r.layer_names.iter().zip(&r.layer_lens) { ref_layers.push((n.clone(), rlay[off..off + len].to_vec())); off += len; }
        anyhow::ensure!(r.t0 == pt0, "xtp: reference t0 {} != this run's {pt0}", r.t0);
        let (lm, lx, hm, hx, div, per) = compare(&ref_lg, &ref_layers, &r.seq, plen, &lg1, &lay1, &am1, pt0);
        let rel = |x: f64, f: f64| x / f.max(1e-30);
        let flips = router_flips(&ref_layers, &lay1, model.cfg.num_experts, model.cfg.num_experts_per_tok);
        crate::dprintln!("XTP router top-{} flips at cmp TP={world} vs TP=1 (layer, ids, ref gap k|k+1): {:?}", model.cfg.num_experts_per_tok, flips);
        crate::dprintln!("XTP floors (TP=1): bit-0 GDN-state logits med {:.3e} hidden med {:.3e} | moe-ulp (reduction site) logits med {:.3e} hidden med {:.3e}",
                 r.floor_logits_med, r.floor_hidden_med, r.moeulp_logits_med, r.moeulp_hidden_med);
        crate::dprintln!("XTP TP={world} vs the bit-0 floor: logits {:.2}x hidden {:.2}x | vs the moe-ulp floor: logits {:.2}x hidden {:.2}x",
                 rel(lm, r.floor_logits_med), rel(hm, r.floor_hidden_med), rel(lm, r.moeulp_logits_med), rel(hm, r.moeulp_hidden_med));
        let mut bl: Vec<f64> = r.band.iter().map(|b| b.2).collect();
        let band_l = median(&mut bl);
        let band_h = r.band.iter().map(|b| b.3).fold(0f64, f64::max);
        crate::dprintln!("XTP band ({} floor members): logits med {:.3e} | hidden med max {:.3e} | member first flips {:?}",
                 r.band.len(), band_l, band_h, r.band.iter().map(|b| (b.0.clone(), b.1, b.4)).collect::<Vec<_>>());
        crate::dprintln!("XTP TP={world} vs the band: logits {:.2}x (<= 3x) | hidden {:.2}x of the band edge (<= 3x) | TP={world} first flip {:?}",
                 rel(lm, band_l), rel(hm, band_h), flips.first());
        let tier_ok = !r.band.is_empty() && lm <= 3.0 * band_l && hm <= 3.0 * band_h;
        let gdiv = (0..gen).find(|&i| seq2[plen + i] != r.seq[plen + i]);
        crate::dprintln!("XTP per-layer hidden relL2 at the last prompt position (TP={world} vs TP=1):");
        for chunk in per.iter().filter(|(n, _)| n.starts_with("layer")).collect::<Vec<_>>().chunks(8) {
            crate::dprintln!("  {}", chunk.iter().map(|(n, v)| format!("{n}={v:.2e}")).collect::<Vec<_>>().join(" "));
        }
        let yatt: Vec<&(String, f64)> = per.iter().filter(|(n, _)| n.starts_with("yatt")).collect();
        if !yatt.is_empty() {
            crate::dprintln!("XTP per-layer mixer-output (yatt, post all-reduce) relL2 at cmp: {}",
                     yatt.iter().take(12).map(|(n, v)| format!("{n}={v:.2e}")).collect::<Vec<_>>().join(" "));
        }
        crate::dprintln!("XTP TP={world} vs TP=1 (teacher-forced, {} steps): logits relL2 med {lm:.3e} ({:.2}x floor) max {lx:.3e} | hidden relL2 med {hm:.3e} ({:.2}x floor) max {hx:.3e} | first argmax divergence {:?} (floor {:?})",
                 total_steps, rel(lm, r.floor_logits_med), rel(hm, r.floor_hidden_med),
                 div.map(|d| d as i64 - (plen as i64 - 1)), r.floor_div.map(|d| d as i64 - (plen as i64 - 1)));
        let text = tok.decode(&seq2[plen..], false).unwrap_or_default();
        crate::dprintln!("XTP greedy[{gen}] TP={world}: {:?}", text);
        crate::dprintln!("XTP greedy[{gen}] TP=1: {:?}", r.text);
        crate::dprintln!("XTP greedy divergence index (free-running TP={world} vs TP=1): {}",
                 gdiv.map(|i| i.to_string()).unwrap_or_else(|| format!("none — all {gen} tokens identical")));
        crate::dprintln!("RESULT: {} (tier: logits relL2 med <= 3x band median {band_l:.3e}, hidden relL2 med <= 3x band edge {band_h:.3e}; {} floor members) ; {}{}",
                 if tier_ok { "XTP_TIER_OK" } else { "XTP_TIER_FAIL" }, r.band.len(),
                 if ident_ok { "RANK_IDENTITY_OK" } else { "RANK_IDENTITY_FAIL" },
                 if opts.gates > 0 { if gates_ok { " ; GATES_OK" } else { " ; GATES_FAIL" } } else { "" });
        if !(tier_ok && ident_ok && gates_ok) { anyhow::bail!("xtp gate failed"); }
    } else {
        if rank == 0 {
            crate::dprintln!("XTP greedy[{gen}] TP={world} (free-running, agreed every step): {:?}", tok.decode(&seq2[plen..], false).unwrap_or_default());
        }
        crate::dprintln!("RESULT: {} ; {}{}", if opts.ident_only { "IDENT_ONLY (no TP=1 reference)" } else { "node (the head holds the reference)" },
                 if ident_ok { "RANK_IDENTITY_OK" } else { "RANK_IDENTITY_FAIL" },
                 if opts.gates > 0 { if gates_ok { " ; GATES_OK" } else { " ; GATES_FAIL" } } else { "" });
        if !(ident_ok && gates_ok) { anyhow::bail!("rank identity / gates FAILED on rank {rank}"); }
    }
    Ok(())
}
