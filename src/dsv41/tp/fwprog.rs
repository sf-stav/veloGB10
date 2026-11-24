//! DSV4.1 TP (leg FW): the serving engine's gates and tables, as parts of the TP model program
//! (`--probe-dsv41-tp --dsv41-tp-program model --dsv41-tp-run <parts>`), every rank running the same program:
//! * `lanes` — batch invariance across lanes: four prompts prefilled on lanes 0-3, then greedy decode steps with the
//!             four lanes in ONE call (4 rows) and the same lanes each ALONE (1 row): every head output (greedy, lse,
//!             max logit) bit-identical; then verify-shaped calls (4 lanes x 6 rows = 24 rows) against each lane's
//!             6-row call alone. RESULT line `FW lanes: ... PASS|FAIL`. Then the same at a LONG context (leg FIXB,
//!             `lanes_long`): lane 0 on the first fixture prompt with P >= 1,000 (the top-512 indexer prunes), the
//!             other lanes short, CED prefills, decode / verify-shaped calls mixed vs alone (eager, captured,
//!             replayed) and lane 0's widths 1..24. RESULT line `FW lanes, long context: ... PASS|FAIL`.
//! * `spec`  — DSpark at TP: greedy speculative == greedy plain, token for token, `--dsv41-tp-gen-tokens` tokens on the
//!             short prompt and on the 4K agent fixture's prompt; then acceptance per workload (prose / code / agent)
//!             at THE POSTURE's sampling (temperature 1, top_p 1, thinking on) and greedy, one lane.
//! * `bench` — the step-time and tok/s table: 1 and 4 lanes, greedy and sampled, speculative (the policy's length) and
//!             plain, with acceptance; decode step times by rows.
//! Every step passes the per-step agree guard (all ranks emitted the same tokens).

use super::fwd::{HeadWant, TpModel, MAX_LANES};
use super::prog::{Guard, ModelOpts};
use super::spec::TpSpec;
use crate::dsv41::ced::schedule::Mode;
use crate::dsv41::dump::RefbDump;
use crate::tokenizer::QwenTokenizer;
use anyhow::{Context, Result};

/// The workloads of the acceptance / speed tables: (name, user message); the agent turn is the 4K fixture's prompt.
pub const CODE_PROMPT: &str = "Write a Python module implementing a thread-safe LRU cache with per-entry TTL expiry: a class with get, put, delete \
    and stats methods, type hints and docstrings, then pytest unit tests covering eviction order, expiry and concurrent access.";
pub const PROSE_PROMPT: &str = super::prog::DEFAULT_PROMPT;
const LANE_PROMPTS: [&str; 4] = [
    "Explain how a refrigerator moves heat out of its interior, step by step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List ten practical tips for writing maintainable Rust code, with a sentence on each.",
    "What were the main causes of the fall of the Western Roman Empire? Answer in a few paragraphs.",
];

/// Chat-encode one user message (thinking on / off) to token ids.
pub fn encode(tok: &QwenTokenizer, msg: &str, thinking: bool) -> Result<Vec<u32>> {
    use crate::dsv41::chat::{jx, render};
    let msgs = vec![jx::obj(vec![("role", jx::s("user")), ("content", jx::s(msg))])];
    let mode = if thinking { render::ThinkingMode::Thinking } else { render::ThinkingMode::Chat };
    let enc = render::encode_messages(&msgs, &render::EncodeOptions::new(mode)).map_err(|e| anyhow::anyhow!("{e}"))?;
    tok.encode(&enc.prompt, false)
}

/// The 4K agent fixture's prompt (rank 0 reads it; the other ranks receive it).
fn agent_prompt(m: &mut TpModel, o: &ModelOpts) -> Result<Vec<u32>> {
    let mine = if m.rank == 0 {
        let fx = o.fixtures.iter().find(|f| RefbDump::open(f).ok().and_then(|d| d.prompt_len().ok()).map_or(false, |p| p >= 1000))
            .context("the agent workload needs a fixture with P >= 1,000 in --dsv41-tp-fixtures (read on rank 0 only)")?;
        let d = RefbDump::open(fx)?;
        Some(d.global("prompt_ids")?.i.iter().map(|&x| x as u32).collect::<Vec<u32>>())
    } else { None };
    m.bcast_tokens(mine)
}

/// Prefill lane `lane` with `toks` through the served prefill (CE's exact CED by default: `--dsv41-prefill` /
/// `--dsv41-ced-keep`), the DSpark taps of the prompt tail when `spec` is given. Returns (seconds, the final stream's
/// row of the last prompt row).
pub fn prefill(m: &mut TpModel, spec: Option<&TpSpec>, lane: usize, toks: &[u32], chunk: usize, mode: Mode) -> Result<(f64, usize)> {
    m.reset_lane_n(lane)?;
    let taps = spec.map(|s| s.taps_dst(m));
    let r = m.prefill(lane, toks, chunk, mode, taps, false)?;
    Ok((r.secs, r.last_row))
}

/// Greedy plain decode of lane 0 from its prefilled prompt (TPC's head exchange, 1 row per step): the first token
/// from the last prompt row, then `n - 1` steps or a stop token.
fn plain_greedy(m: &mut TpModel, guard: &mut Guard, lane: usize, toks: &[u32], chunk: usize, n: usize, stops: &[u32], mode: Mode) -> Result<(Vec<u32>, f64)> {
    let (_, last) = prefill(m, None, lane, toks, chunk, mode)?;
    let first = m.head_last(last)?.greedy as u32;
    guard.agree("plain prefill", first)?;
    let mut out = vec![first];
    let t = std::time::Instant::now();
    let mut cur = first;
    while out.len() < n && !stops.contains(&cur) {
        crate::thermal::pace(); // NIGHT1: once per step on every rank (timing only)
        let o = m.forward_segs(&[(lane, &[cur])], HeadWant::Last, None)?;
        cur = o[0].greedy as u32;
        guard.agree("plain step", cur ^ ((m.lanes[lane].pos as u32) << 20))?;
        out.push(cur);
    }
    let secs = t.elapsed().as_secs_f64();
    Ok((out.clone(), (out.len().saturating_sub(1)) as f64 / secs.max(1e-9)))
}

/// Speculative decode of the lanes `lanes` (each with its prompt), `n` tokens per lane or a stop token. Returns the
/// tokens per lane and the decode seconds (after the prefills).
#[allow(clippy::too_many_arguments)]
fn spec_decode(m: &mut TpModel, sp: &mut TpSpec, guard: &mut Guard, prompts: &[Vec<u32>], chunk: usize, n: usize, stops: &[u32], temp: f32, seed0: u64,
               k_pin: Option<usize>, pack: &std::path::Path, mode: Mode) -> Result<(Vec<Vec<u32>>, f64, f64)> {
    let nl = prompts.len();
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); nl];
    let tp = std::time::Instant::now();
    for (l, p) in prompts.iter().enumerate() {
        m.reset_lane_n(l)?;
        sp.reset_lane(m.f.g, l)?;
        let (_, last) = prefill(m, Some(sp), l, p, chunk, mode)?;
        let t0 = sp.start_lane(m, l, p.len(), last, temp, seed0 + l as u64)?;
        guard.agree("spec first token", t0)?;
        outs[l].push(t0);
        if stops.contains(&t0) { sp.host[l].active = 0; }
    }
    let pre_s = tp.elapsed().as_secs_f64();
    if k_pin.is_none() && !CALIBRATED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let line = sp.calibrate(m, nl, pack)?;
        guard.agree("calibration", 0x5CA1)?;
        CALIB_LINE.lock().map(|mut l| *l = line).ok();
    }
    let t = std::time::Instant::now();
    loop {
        for l in 0..nl { if outs[l].len() >= n || outs[l].last().map_or(false, |t| stops.contains(t)) { sp.host[l].active = 0; } }
        if (0..nl).all(|l| sp.host[l].active == 0) { break; }
        crate::thermal::pace(); // NIGHT1: once per round on every rank (timing only)
        let k = match k_pin { Some(k) => k, None => sp.choose_k(m, nl)? };
        let em = sp.round(m, nl, k)?;
        let mut hsh = 0u32;
        for (l, e) in em.iter().enumerate() {
            for &tk in e {
                if outs[l].len() < n && !outs[l].last().map_or(false, |t| stops.contains(t)) { outs[l].push(tk); }
                hsh = hsh.wrapping_mul(31) ^ tk;
            }
        }
        guard.agree("spec round", hsh)?;
    }
    Ok((outs, t.elapsed().as_secs_f64(), pre_s))
}

/// FNV-1a over device bytes (downloaded).
fn dev_hash(m: &TpModel, ptr: u64, bytes: usize) -> Result<u64> {
    let v = m.f.g.down::<u8>(ptr, bytes)?;
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in v { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3); }
    Ok(h)
}

/// The decode state of lane `lane` after a `p`-row prefill, as hashes: every layer's ring slots of the positions a
/// decode can read before overwriting them (the last 128: window 128; the 6 backtracking slots of positions below are
/// overwritten first), every KV source's caches up to its groups (main KV, index-K planes) and its frontier ring.
fn state_hashes(m: &TpModel, lane: usize, p: usize) -> Result<Vec<(String, u64)>> {
    let cfg = m.cfg;
    let (hd, slots) = (cfg.head_dim, cfg.swa_slots());
    let ln = &m.lanes[lane];
    let mut v = Vec::new();
    for l in 0..cfg.layers {
        let w = &ln.win[l];
        let (mut hq, mut hs) = (0u64, 0u64);
        for pos in p.saturating_sub(cfg.window)..p {
            let sl = pos % slots;
            hq = hq.rotate_left(7) ^ dev_hash(m, w.ring_q.ptr() + (sl * hd) as u64, hd)?;
            hs = hs.rotate_left(7) ^ dev_hash(m, w.ring_s.ptr() + (sl * hd / 32) as u64, hd / 32)?;
        }
        v.push((format!("L{l:02} ring"), hq ^ hs.rotate_left(32)));
    }
    let mut srcs: Vec<_> = ln.src.keys().copied().collect();
    srcs.sort_unstable();
    for s in srcs {
        let st = &ln.src[&s];
        let n = st.n_groups;
        v.push((format!("src L{s:02} groups"), n as u64));
        v.push((format!("src L{s:02} main KV"), dev_hash(m, st.kv_main.ptr(), n * 288)?));
        v.push((format!("src L{s:02} index K"), dev_hash(m, st.iq.ptr(), n * 64)? ^ dev_hash(m, st.is.ptr(), n * 4)?.rotate_left(17)));
        if let Some((a, b)) = &st.front { v.push((format!("src L{s:02} frontier"), dev_hash(m, a.ptr(), a.bytes)? ^ dev_hash(m, b.ptr(), b.bytes)?.rotate_left(29))); }
    }
    Ok(v)
}

/// CE's gate at TP: on each long fixture prompt (P >= 1,000; rank 0 reads it), the every-layer prefill (Exact) and the
/// CED prefill at the program's chunk and at a second chunking: the decode state (`state_hashes`), the DSpark taps of
/// the prompt tail and the first-token head exchange (greedy id, log-sum, max logit) bit for bit.
fn ced_gate(m: &mut TpModel, guard: &mut Guard, o: &ModelOpts, chunk: usize, say: &mut dyn FnMut(String)) -> Result<bool> {
    let mut ok = true;
    let fx: Vec<std::path::PathBuf> = o.fixtures.clone();
    let tld = m.cfg.dspark_targets.len() * m.cfg.hidden;
    let taps = m.f.g.buf(super::spec::TAP_ROWS * tld * 2)?;
    for f in fx {
        let mine = if m.rank == 0 {
            let d = RefbDump::open(&f)?;
            let ids: Vec<u32> = d.global("prompt_ids")?.i.iter().map(|&x| x as u32).collect();
            Some(if ids.len() >= 1000 { ids } else { Vec::new() })
        } else { None };
        let ids = m.bcast_tokens(mine)?;
        if ids.is_empty() { continue; }
        let p = ids.len();
        anyhow::ensure!(p + 8 <= o.max_seq, "fixture {} (P {p}) needs --max-seq-len > {}", f.display(), p + 8);
        let mut runs: Vec<(String, f64, Vec<(String, u64)>, u64, [u32; 3])> = Vec::new();
        let c2 = if chunk > 1001 { 1001 } else { chunk + 7 };
        for (name, mode, ch) in [("exact", Mode::Exact, chunk), ("ced", Mode::Ced, chunk), ("ced", Mode::Ced, c2)] {
            m.reset_lane_n(0)?;
            m.f.g.memset(taps.ptr(), taps.bytes, 0)?;
            let r = m.prefill(0, &ids, ch, mode, Some((taps.ptr(), tld)), true)?;
            let st = state_hashes(m, 0, p)?;
            let th = dev_hash(m, taps.ptr(), p.min(super::spec::TAP_ROWS) * tld * 2)?;
            let hd = r.head.context("no head")?;
            let hb = [hd.greedy as u32, hd.lse.to_bits(), hd.max_x.to_bits()];
            let all = st.iter().fold(th ^ hb[0] as u64, |a, (_, h)| a.rotate_left(5) ^ h);
            guard.agree("ced state", (all ^ (all >> 32)) as u32)?;
            say(format!("  {} {name}@{ch}: prefill {:.2} s = {:.0} tok/s, layer-rows {} of {} ({:.3}), first token {} (lse {:.4})", f.display(), r.secs,
                        p as f64 / r.secs, r.layer_rows, r.exact_rows, r.layer_rows as f64 / r.exact_rows as f64, hd.greedy, hd.lse));
            runs.push((format!("{name}@{ch}"), r.secs, st, th, hb));
        }
        // prefix resume == cold: the CED prefill of the prompt's first k rows (a Full state at its end), snapshot, restore
        // into a fresh lane, resume the whole prompt from k (CED, the program's chunk)
        {
            let k = p - p / 7;
            m.reset_lane_n(0)?;
            m.prefill(0, &ids[..k], chunk, Mode::Ced, None, false)?;
            let snap = m.snapshot(0)?;
            m.reset_lane_n(0)?;
            m.restore(0, &snap)?;
            m.f.g.memset(taps.ptr(), taps.bytes, 0)?;
            let r = m.prefill(0, &ids, chunk, Mode::Ced, Some((taps.ptr(), tld)), true)?;
            let st = state_hashes(m, 0, p)?;
            let th = dev_hash(m, taps.ptr(), p.min(super::spec::TAP_ROWS) * tld * 2)?;
            let hd = r.head.context("no head")?;
            let hb = [hd.greedy as u32, hd.lse.to_bits(), hd.max_x.to_bits()];
            let all = st.iter().fold(th ^ hb[0] as u64, |a, (_, h)| a.rotate_left(5) ^ h);
            guard.agree("ced resume state", (all ^ (all >> 32)) as u32)?;
            say(format!("  {} resume: state of {k} rows ({:.1} MB snapshot, {:?}), resumed prefill of the last {} rows {:.2} s, layer-rows {}", f.display(),
                        snap.bytes as f64 / 1e6, snap.kind, p - k, r.secs, r.layer_rows));
            runs.push((format!("ced:resume@{k}"), r.secs, st, th, hb));
        }
        // decode at this context from the prefilled state: eager vs CUDA graphs, bit for bit (the long-context plan class)
        {
            let snap = m.snapshot(0)?;
            let first = runs.last().map(|r| r.4[0]).unwrap_or(0);
            let steps = 16usize;
            let mut res: Vec<(Vec<[u32; 3]>, f64)> = Vec::new();
            for graphs_on in [false, true] {
                if !graphs_on && m.graphs.is_none() { continue; }
                let saved = if graphs_on { None } else { m.graphs.take() };
                m.restore(0, &snap)?;
                let mut cur = first;
                let mut v = Vec::new();
                let t = std::time::Instant::now();
                for _ in 0..steps {
                    let o1 = m.forward_segs(&[(0, &[cur])], HeadWant::Last, None)?;
                    v.push([o1[0].greedy as u32, o1[0].lse.to_bits(), o1[0].max_x.to_bits()]);
                    cur = o1[0].greedy as u32;
                    guard.agree("long-context decode", cur)?;
                }
                res.push((v, t.elapsed().as_secs_f64() * 1e3 / steps as f64));
                if let Some(g) = saved { m.graphs = Some(g); }
            }
            if res.len() == 2 {
                let diff = res[0].0.iter().zip(&res[1].0).filter(|(a, b)| a != b).count();
                say(format!("FW long-context decode, {} (context {p}): {steps} greedy steps eager {:.1} ms/step vs CUDA graphs {:.1} ms/step: {diff} of {steps} head outputs differ -> {}",
                            f.display(), res[0].1, res[1].1, if diff == 0 { "PASS" } else { "FAIL" }));
                ok &= diff == 0;
            }
        }
        let (ref_name, _, ref_st, ref_th, ref_hb) = runs[0].clone();
        for (name, _, st, th, hb) in &runs[1..] {
            let diffs: Vec<&str> = st.iter().zip(ref_st.iter()).filter(|(a, b)| a.1 != b.1).map(|(a, _)| a.0.as_str()).collect();
            let same = diffs.is_empty() && *th == ref_th && *hb == ref_hb;
            say(format!("FW ced gate, {} (P {p}): {name} vs {ref_name}: decode state {} of {} items equal{}, DSpark taps {}, first-token head {} -> {}",
                        f.display(), st.len() - diffs.len(), st.len(), if diffs.is_empty() { String::new() } else { format!(" (differ: {})", diffs.join(", ")) },
                        if *th == ref_th { "equal" } else { "DIFFER" }, if *hb == ref_hb { "equal" } else { "DIFFERS" }, if same { "PASS" } else { "FAIL" }));
            ok &= same;
        }
    }
    m.reset_lane_n(0)?;
    Ok(ok)
}

/// Batch invariance where the indexer prunes (leg FIXB; the `lanes` part above runs ~20-token contexts, where every
/// top-512 selection keeps every key): lane 0 holds the first fixture prompt with P >= 1,000 (`--dsv41-tp-fixtures`,
/// read on rank 0), the other lanes short chat prompts, every lane prefilled through the served CED prefill and
/// snapshotted. From those states: (a) 32 greedy decode steps with every lane in ONE call vs each lane ALONE;
/// (b) verify-shaped calls, 6 rows per lane in one call vs each lane's 6-row call, three times (a shape's first call
/// runs eagerly, the second is captured, the third replays the graph); (c) lane 0 at its long context: widths 1..24
/// against its 24-row call (widths 1..6 twice more: captured, replayed). Every head output (greedy id, log-sum, max
/// logit) must be bit-identical. RESULT line `FW lanes, long context: ... PASS|FAIL` (SKIPPED without such a fixture
/// or with too small a `--max-seq-len`).
fn lanes_long(m: &mut TpModel, guard: &mut Guard, o: &ModelOpts, tok: &QwenTokenizer, chunk: usize, mode: Mode, say: &mut dyn FnMut(String)) -> Result<bool> {
    m.barrier(0x5180)?;
    let nl = m.lanes.len().min(MAX_LANES);
    let mine = if m.rank == 0 {
        let ids = o.fixtures.iter().find_map(|f| {
            let d = RefbDump::open(f).ok()?;
            if d.prompt_len().ok()? < 1000 { return None; }
            Some(d.global("prompt_ids").ok()?.i.iter().map(|&x| x as u32).collect::<Vec<u32>>())
        });
        Some(ids.unwrap_or_default())
    } else { None };
    let long = m.bcast_tokens(mine)?;
    let (steps, vr) = (32usize, 6usize);
    if long.is_empty() || long.len() + steps + 64 > o.max_seq {
        say(format!("FW lanes, long context: SKIPPED ({})", if long.is_empty() { "no fixture with P >= 1,000 in --dsv41-tp-fixtures".to_string() }
                    else { format!("P {} needs --max-seq-len > {}", long.len(), long.len() + steps + 64) }));
        return Ok(true);
    }
    let p = long.len();
    let bits = |x: &super::comm::TpOut| [x.greedy as u32, x.lse.to_bits(), x.max_x.to_bits()];
    let mut prompts = vec![long];
    for l in 1..nl { prompts.push(encode(tok, LANE_PROMPTS[l % LANE_PROMPTS.len()], true)?); }
    let mut first = vec![0u32; nl];
    let mut snaps = Vec::with_capacity(nl);
    for l in 0..nl {
        m.reset_lane_n(l)?;
        let r = m.prefill(l, &prompts[l], chunk, mode, None, true)?;
        first[l] = r.head.context("the long-context lanes prefill returned no head")?.greedy as u32;
        guard.agree("lanes-long prefill", first[l])?;
        snaps.push(m.snapshot(l)?);
    }
    // (a) decode rows: all lanes in one call vs each lane alone, from the same snapshotted states
    for l in 0..nl { m.restore(l, &snaps[l])?; }
    let mut seq: Vec<Vec<u32>> = first.iter().map(|&t| vec![t]).collect();
    let mut a_bits: Vec<Vec<[u32; 3]>> = vec![Vec::new(); nl];
    for _ in 0..steps {
        let cur: Vec<[u32; 1]> = (0..nl).map(|l| [*seq[l].last().unwrap()]).collect();
        let parts: Vec<(usize, &[u32])> = (0..nl).map(|l| (l, cur[l].as_slice())).collect();
        let outs = m.forward_segs(&parts, HeadWant::Last, None)?;
        let mut h = 0u32;
        for l in 0..nl { seq[l].push(outs[l].greedy as u32); a_bits[l].push(bits(&outs[l])); h = h.wrapping_mul(31) ^ outs[l].greedy as u32; }
        guard.agree("lanes-long mixed step", h)?;
    }
    let mut diff = 0usize;
    for l in 0..nl {
        m.restore(l, &snaps[l])?;
        for s in 0..steps {
            let o1 = m.forward_segs(&[(l, &[seq[l][s]])], HeadWant::Last, None)?;
            if bits(&o1[0]) != a_bits[l][s] { diff += 1; }
            guard.agree("lanes-long alone step", o1[0].greedy as u32)?;
        }
    }
    // (b) verify-shaped rows: every lane is at its prompt + 32 positions; roll back 6 and feed the 6 tokens those
    // positions held, all lanes in one call vs each alone, three times (eager, captured, replayed)
    let feed: Vec<Vec<u32>> = (0..nl).map(|l| seq[l][seq[l].len() - 1 - vr..seq[l].len() - 1].to_vec()).collect();
    let mut vdiff = 0usize;
    for pass in 0..3u32 {
        for l in 0..nl { m.rollback(l, m.lanes[l].pos - vr)?; }
        let parts: Vec<(usize, &[u32])> = (0..nl).map(|l| (l, feed[l].as_slice())).collect();
        let mixed: Vec<[u32; 3]> = m.forward_segs(&parts, HeadWant::All, None)?.iter().map(bits).collect();
        guard.agree("lanes-long verify mixed", mixed.iter().fold(pass, |a, x| a.wrapping_mul(31) ^ x[0]))?;
        for l in 0..nl {
            m.rollback(l, m.lanes[l].pos - vr)?;
            let o1 = m.forward_segs(&[(l, feed[l].as_slice())], HeadWant::All, None)?;
            for (j, x) in o1.iter().enumerate() { if bits(x) != mixed[l * vr + j] { vdiff += 1; } }
            guard.agree("lanes-long verify alone", o1.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
        }
    }
    // (c) lane 0's widths at the long context against its 24-row call
    let wt: Vec<u32> = (0..24).map(|i| 1000 + ((i * 37) % 50_000) as u32).collect();
    m.restore(0, &snaps[0])?;
    let reference: Vec<[u32; 3]> = m.forward_segs(&[(0, &wt[..])], HeadWant::All, None)?.iter().map(bits).collect();
    guard.agree("lanes-long width reference", reference.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x[0]))?;
    let (mut wrows, mut wbad) = (0usize, Vec::new());
    let widths: Vec<usize> = (1..=24).chain(1..=6).chain(1..=6).collect();
    for &w in &widths {
        m.restore(0, &snaps[0])?;
        let outs = m.forward_segs(&[(0, &wt[..w])], HeadWant::All, None)?;
        for (r, x) in outs.iter().enumerate() { wrows += 1; if bits(x) != reference[r] { wbad.push(format!("w{w}r{r}")); } }
        guard.agree("lanes-long width call", outs.iter().fold(w as u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
    }
    for l in 0..nl { m.reset_lane_n(l)?; }
    let pass = diff == 0 && vdiff == 0 && wbad.is_empty();
    let text = tok.decode(&seq[0], false).unwrap_or_default().replace('\n', " ");
    say(format!("  long lane 0 (P {p}): <<<{}>>>", text.chars().take(160).collect::<String>()));
    say(format!("FW lanes, long context: lane 0 at P {p} (the indexer prunes) + {} short lanes: decode {diff} of {} head outputs differ (mixed {nl}-row calls vs alone); \
                 verify-shaped {vdiff} of {} differ ({nl} x {vr} rows in one call vs alone, 3 passes: eager, captured, replayed); lane 0 widths 1-24 (+1-6 twice) \
                 at context {p}: {} of {wrows} rows differ{} -> {}", nl - 1, nl * steps, 3 * nl * vr, wbad.len(),
                if wbad.is_empty() { String::new() } else { format!(" ({})", wbad.iter().take(8).cloned().collect::<Vec<_>>().join(" ")) },
                if pass { "PASS" } else { "FAIL" }));
    Ok(pass)
}

static CALIBRATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static CALIB_LINE: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Run the FW parts named in `o.run`. Returns whether every gate held.
#[allow(clippy::too_many_arguments)]
pub fn run(m: &mut TpModel, sp: Option<&mut TpSpec>, guard: &mut Guard, tok: &QwenTokenizer, stops: &[u32], o: &ModelOpts, say: &mut dyn FnMut(String)) -> Result<bool> {
    let mut ok = true;
    let chunk = o.chunk;
    let mode = crate::dsv41::ced::schedule::mode_from_flags()?;
    say(format!("FW prefill: {} in chunks of {chunk} rows (wide row family for every prefill linear)", mode.name()));
    // ------------------------------------------------------------------ ced
    if o.run.iter().any(|r| r == "ced") {
        m.barrier(0x5400)?;
        ok &= ced_gate(m, guard, o, chunk, say)?;
        m.settle()?;
    }
    // ------------------------------------------------------------------ lanes
    if o.run.iter().any(|r| r == "lanes") {
        m.barrier(0x5100)?;
        let nl = m.lanes.len().min(MAX_LANES);
        anyhow::ensure!(nl >= 2, "--dsv41-tp-run lanes needs --dsv41-tp-lanes >= 2");
        let prompts: Vec<Vec<u32>> = LANE_PROMPTS.iter().take(nl).map(|p| encode(tok, p, true)).collect::<Result<_>>()?;
        let steps = 32usize;
        let bits = |o: &super::comm::TpOut| [o.greedy as u32, o.lse.to_bits(), o.max_x.to_bits()];
        // A: every lane prefilled, then 4-row mixed calls
        let mut seq: Vec<Vec<u32>> = vec![Vec::new(); nl];
        let mut a_bits: Vec<Vec<[u32; 3]>> = vec![Vec::new(); nl];
        for l in 0..nl {
            m.reset_lane_n(l)?;
            let p = &prompts[l];
            let mut r0 = 0;
            while r0 < p.len() {
                let c = (p.len() - r0).min(chunk);
                let outs = m.forward_segs(&[(l, &p[r0..r0 + c])], if r0 + c == p.len() { HeadWant::Last } else { HeadWant::None }, None)?;
                if let Some(x) = outs.last() { seq[l].push(x.greedy as u32); }
                r0 += c;
            }
        }
        let t = std::time::Instant::now();
        for _ in 0..steps {
            let cur: Vec<[u32; 1]> = (0..nl).map(|l| [*seq[l].last().unwrap()]).collect();
            let parts: Vec<(usize, &[u32])> = (0..nl).map(|l| (l, cur[l].as_slice())).collect();
            let outs = m.forward_segs(&parts, HeadWant::Last, None)?;
            let mut h = 0u32;
            for l in 0..nl { seq[l].push(outs[l].greedy as u32); a_bits[l].push(bits(&outs[l])); h = h.wrapping_mul(31) ^ outs[l].greedy as u32; }
            guard.agree("lanes mixed step", h)?;
        }
        let mixed_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        // B: each lane alone (the same tokens fed)
        let mut diff = 0usize;
        let mut alone_ms = 0.0;
        for l in 0..nl {
            m.reset_lane_n(l)?;
            let p = &prompts[l];
            let mut r0 = 0;
            while r0 < p.len() { let c = (p.len() - r0).min(chunk); m.forward_segs(&[(l, &p[r0..r0 + c])], HeadWant::None, None)?; r0 += c; }
            let t = std::time::Instant::now();
            for s in 0..steps {
                let o1 = m.forward_segs(&[(l, &[seq[l][s]])], HeadWant::Last, None)?;
                if bits(&o1[0]) != a_bits[l][s] { diff += 1; }
                guard.agree("lanes alone step", o1[0].greedy as u32)?;
            }
            alone_ms += t.elapsed().as_secs_f64() * 1e3 / steps as f64 / nl as f64;
        }
        say(format!("FW lanes, decode rows: {nl} lanes x {steps} greedy steps, mixed {nl}-row calls ({mixed_ms:.1} ms/step) vs each lane alone (1 row, {alone_ms:.1} ms/step): \
                     {diff} of {} head outputs differ (greedy id, lse, max logit bits)", nl * steps));
        ok &= diff == 0;
        // C: verify-shaped: 6 rows per lane, all lanes in one call (24 rows) vs each lane alone (6 rows), from the same state
        let vr = 6usize;
        let mut vdiff = 0usize;
        let mut mixed = Vec::new();
        for l in 0..nl { m.rollback(l, m.lanes[l].pos - vr.min(steps))?; }
        let feed: Vec<Vec<u32>> = (0..nl).map(|l| seq[l][seq[l].len() - 1 - vr..seq[l].len() - 1].to_vec()).collect();
        // (the lanes are at pos - 6: feed the six tokens those positions held)
        let parts: Vec<(usize, &[u32])> = (0..nl).map(|l| (l, feed[l].as_slice())).collect();
        let outs = m.forward_segs(&parts, HeadWant::All, None)?;
        for x in &outs { mixed.push(bits(x)); }
        guard.agree("lanes verify mixed", outs.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
        for l in 0..nl {
            m.rollback(l, m.lanes[l].pos - vr)?;
            let o1 = m.forward_segs(&[(l, feed[l].as_slice())], HeadWant::All, None)?;
            for (j, x) in o1.iter().enumerate() { if bits(x) != mixed[l * vr + j] { vdiff += 1; } }
            guard.agree("lanes verify alone", o1.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
        }
        say(format!("FW lanes, verify rows: {nl} lanes x {vr} rows in one {}-row call vs each lane's {vr}-row call alone: {vdiff} of {} head outputs differ",
                    nl * vr, nl * vr));
        ok &= vdiff == 0;
        let texts: Vec<String> = (0..nl).map(|l| tok.decode(&seq[l], false).unwrap_or_default().replace('\n', " ")).collect();
        for (l, t) in texts.iter().enumerate() { say(format!("  lane {l}: <<<{}>>>", t.chars().take(160).collect::<String>())); }
        say(format!("FW lanes: {}", if diff == 0 && vdiff == 0 { "PASS" } else { "FAIL" }));
        ok &= lanes_long(m, guard, o, tok, chunk, mode, say)?;
        m.settle()?;
    }
    // ------------------------------------------------------------------ graphs: eager == graph replay, bit for bit
    if o.run.iter().any(|r| r == "graphs") && m.graphs.is_some() {
        m.barrier(0x5150)?;
        let nl = m.lanes.len().min(MAX_LANES);
        let prompts: Vec<Vec<u32>> = LANE_PROMPTS.iter().take(nl).map(|p| encode(tok, p, true)).collect::<Result<_>>()?;
        let steps = 48usize;
        let bits = |o: &super::comm::TpOut| [o.greedy as u32, o.lse.to_bits(), o.max_x.to_bits()];
        let mut runs: Vec<Vec<Vec<[u32; 3]>>> = Vec::new();
        let mut times = Vec::new();
        for graphs_on in [false, true] {
            let saved = if graphs_on { None } else { m.graphs.take() };
            let mut per: Vec<Vec<[u32; 3]>> = vec![Vec::new(); nl];
            let mut cur: Vec<u32> = vec![0; nl];
            for l in 0..nl {
                let (_, last) = prefill(m, None, l, &prompts[l], chunk, mode)?;
                cur[l] = m.head_last(last)?.greedy as u32;
            }
            let t = std::time::Instant::now();
            for st in 0..steps {
                // 1-lane steps for lane 0 in the first half, then all lanes mixed (both call shapes)
                let lanes: Vec<usize> = if st < steps / 2 { vec![0] } else { (0..nl).collect() };
                let cv: Vec<[u32; 1]> = lanes.iter().map(|&l| [cur[l]]).collect();
                let parts: Vec<(usize, &[u32])> = lanes.iter().zip(&cv).map(|(&l, c)| (l, c.as_slice())).collect();
                let outs = m.forward_segs(&parts, HeadWant::Last, None)?;
                for (i, &l) in lanes.iter().enumerate() { per[l].push(bits(&outs[i])); cur[l] = outs[i].greedy as u32; }
                guard.agree("graph gate step", outs.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
            }
            times.push(t.elapsed().as_secs_f64() * 1e3 / steps as f64);
            if let Some(g) = saved { m.graphs = Some(g); }
            runs.push(per);
        }
        let diff: usize = (0..nl).map(|l| runs[0][l].iter().zip(&runs[1][l]).filter(|(a, b)| a != b).count()).sum();
        let n: usize = runs[0].iter().map(|v| v.len()).sum();
        say(format!("FW graph gate: {steps} greedy steps (lane 0 alone, then {nl} lanes mixed), eager {:.1} ms/step vs CUDA graphs {:.1} ms/step: {diff} of {n} head outputs differ -> {}",
                    times[0], times[1], if diff == 0 { "PASS" } else { "FAIL" }));
        ok &= diff == 0;
        m.settle()?;
    }
    let Some(sp) = sp else { return Ok(ok) };
    // ------------------------------------------------------------------ spec
    if o.run.iter().any(|r| r == "spec") {
        m.barrier(0x5200)?;
        let n = o.gen_tokens;
        let agent = agent_prompt(m, o)?;
        for (name, prompt) in [("short (prose essay)", encode(tok, &o.prompt, true)?), ("agent_4k fixture", agent.clone())] {
            let (plain, plain_rate) = plain_greedy(m, guard, 0, &prompt, chunk, n, stops, mode)?;
            let (outs, secs, _) = spec_decode(m, sp, guard, std::slice::from_ref(&prompt), chunk, n, stops, 0.0, 7, None, &o.model_dir, mode)?;
            let s = &outs[0];
            let same = s.iter().zip(plain.iter()).take_while(|(a, b)| a == b).count();
            let equal = s.len() == plain.len() && same == s.len();
            let st = sp.stats[0];
            let fnv = |v: &[u32]| v.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &t| (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01B3));
            say(format!("FW spec gate tokens: plain fnv {:016x}, speculative fnv {:016x}", fnv(&plain), fnv(s)));
            say(format!("FW spec gate, {name} (P {}): greedy speculative {} tokens vs greedy plain {} tokens: {} (first difference at token {same}); \
                         plain {plain_rate:.1} tok/s, speculative {:.1} tok/s; {} rounds, accepted {} of {} drafts ({:.2} tokens / round)",
                        prompt.len(), s.len(), plain.len(), if equal { "IDENTICAL" } else { "DIFFERENT" },
                        (s.len().saturating_sub(1)) as f64 / secs.max(1e-9), st.rounds, st.accepted, st.drafted, st.emitted as f64 / st.rounds.max(1) as f64));
            if !equal { say(format!("  plain:<<<{}>>>\n  spec:<<<{}>>>", tok.decode(&plain, false).unwrap_or_default(), tok.decode(s, false).unwrap_or_default())); }
            ok &= equal;
        }
        m.settle()?;
    }
    // ------------------------------------------------------------------ accept: acceptance per workload
    if o.run.iter().any(|r| r == "accept") {
        m.barrier(0x5250)?;
        let n = o.gen_tokens;
        let agent = agent_prompt(m, o)?;
        // acceptance per workload: sampled (THE POSTURE: temperature 1, top_p 1, thinking on) and greedy
        let work: Vec<(&str, Vec<u32>)> = vec![("prose", encode(tok, PROSE_PROMPT, true)?), ("code", encode(tok, CODE_PROMPT, true)?), ("agent", agent)];
        for (wname, p) in &work {
            for (mname, temp) in [("sampled T1", 1.0f32), ("greedy", 0.0)] {
                let (outs, secs, pre) = spec_decode(m, sp, guard, std::slice::from_ref(p), chunk, n.min(512), stops, temp, 1234, None, &o.model_dir, mode)?;
                let st = sp.stats[0];
                say(format!("FW acceptance, {wname} (P {}), {mname}, 1 lane: {} tokens in {secs:.2} s ({:.1} tok/s; prefill {pre:.2} s), {} rounds, {:.2} tokens / round, \
                             accepted {} of {} drafts ({:.1} %), final draft length {}",
                            p.len(), outs[0].len(), (outs[0].len().saturating_sub(1)) as f64 / secs.max(1e-9), st.rounds, st.emitted as f64 / st.rounds.max(1) as f64,
                            st.accepted, st.drafted, 100.0 * st.accepted as f64 / st.drafted.max(1) as f64, sp.policy.current));
                if *wname == "prose" && temp > 0.0 { say(format!("  <<<{}>>>", tok.decode(&outs[0], false).unwrap_or_default().chars().take(400).collect::<String>())); }
            }
        }
        m.settle()?;
    }
    // ------------------------------------------------------------------ bench
    if o.run.iter().any(|r| r == "bench") {
        m.barrier(0x5300)?;
        let n = o.gen_tokens.min(256);
        let nl_max = m.lanes.len().min(sp.cap);
        let prompts: Vec<Vec<u32>> = LANE_PROMPTS.iter().map(|p| encode(tok, p, true)).collect::<Result<_>>()?;
        say("FW bench (TP forward + DSpark round; host launches unless the graph line says otherwise):".into());
        for &nl in [1usize, nl_max].iter().filter(|&&x| x >= 1) {
            for (mname, temp) in [("sampled T1", 1.0f32), ("greedy", 0.0)] {
                for k_pin in [None, Some(5usize), Some(0usize)] {
                    let ps: Vec<Vec<u32>> = prompts.iter().take(nl).cloned().collect();
                    let (outs, secs, _) = spec_decode(m, sp, guard, &ps, chunk, n, stops, temp, 99, k_pin, &o.model_dir, mode)?;
                    let toks: usize = outs.iter().map(|v| v.len().saturating_sub(1)).sum();
                    let (r, a, d, e) = sp.stats[..nl].iter().fold((0u64, 0u64, 0u64, 0u64), |s, x| (s.0 + x.rounds, s.1 + x.accepted, s.2 + x.drafted, s.3 + x.emitted));
                    say(format!("  {nl} lane(s), {mname}, {}: {toks} tokens in {secs:.2} s = {:.1} tok/s aggregate ({:.1} per lane); {} rounds/lane, {:.2} tokens / round, \
                                 accepted {a} of {d}; last round: draft {:.1} ms, verify {:.1} ms",
                                match k_pin { Some(0) => "plain (k = 0)".to_string(), Some(k) => format!("speculative (k pinned {k})"), None => format!("speculative (policy, final k {})", sp.policy.current) }, toks as f64 / secs, toks as f64 / secs / nl as f64,
                                r / nl as u64, e as f64 / r.max(1) as f64, sp.t_draft, sp.t_verify));
                }
            }
            if nl == nl_max { break; }
        }
        // decode-call step times by rows (one lane / all lanes)
        for (rows_per_lane, lanes) in [(1usize, 1usize), (6, 1), (1, nl_max), (6, nl_max)] {
            let (mut v, mut enq) = (Vec::new(), Vec::new());
            for it in 0..12 {
                let toks: Vec<Vec<u32>> = (0..lanes).map(|l| (0..rows_per_lane).map(|i| 1000 + ((it * 7 + i + l * 13) % 50000) as u32).collect()).collect();
                let parts: Vec<(usize, &[u32])> = (0..lanes).map(|l| (l, toks[l].as_slice())).collect();
                m.f.g.sync()?;
                let t = std::time::Instant::now();
                m.forward_segs(&parts, HeadWant::None, None)?;
                let e = t.elapsed().as_secs_f64() * 1e3;
                m.f.g.sync()?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                guard.agree("bench step", lanes as u32)?;
                for l in 0..lanes { m.rollback(l, m.lanes[l].pos - rows_per_lane)?; }
                if it >= 2 { v.push(ms); enq.push(e); }
            }
            v.sort_by(|a, b| a.total_cmp(b));
            enq.sort_by(|a, b| a.total_cmp(b));
            say(format!("  forward call {lanes} lane(s) x {rows_per_lane} row(s) = {} rows at context ~{} (no head): median {:.2} ms (min {:.2}, max {:.2}); host enqueue median {:.2} ms",
                        lanes * rows_per_lane, m.lanes[0].pos, v[v.len() / 2], v[0], v[v.len() - 1], enq[enq.len() / 2]));
        }
        m.settle()?;
    }
    // ------------------------------------------------------------------ sq1 (leg SQ1): sampling exactness, off-script rate
    if o.run.iter().any(|r| r == "sq1") {
        m.barrier(0x5600)?;
        let b = super::sq1::budget(crate::opts::lookup("dsv41-sq1-budget").and_then(crate::opts::get).filter(|s| s != "unset").as_deref())?;
        ok &= super::sq1::run(m, sp, guard, tok, stops, o, b, say)?;
        m.settle()?;
    }
    if let Ok(l) = CALIB_LINE.lock() { if !l.is_empty() { say(l.clone()); } }
    say(super::metrics::fw_line());
    Ok(ok)
}
