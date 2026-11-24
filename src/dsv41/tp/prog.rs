//! DSV4.1 TP (leg TP): the full-model program every rank runs (`--probe-dsv41-tp --dsv41-tp-program model`).
//! The head ships the shards (TPC's launch), every rank loads its slices (`wload.rs`) into the TP trunk (`fwd.rs`),
//! then runs the parts named by `--dsv41-tp-run` (comma-separated, in this order):
//! * `gate`  — the per-layer gate at TP on REFb's fixtures (`--dsv41-tp-fixtures`; every rank reads them: the
//!             dirs must exist on every box at the same path), SL's EXL3 tolerances (`gate.rs`);
//! * `nll`   — teacher-forced NLL of the long fixture's prompt (the one with P >= 1,000) through the full non-CED
//!             chunked prefill, against REFb's per-row NLL of the native model; then a greedy reply to that prompt;
//! * `gen`   — a greedy reply to `--dsv41-tp-prompt` (chat-encoded, thinking on), up to `--dsv41-tp-gen-tokens`;
//! * `time`  — decode step time at 1 / 6 / 24 rows (the lane's context as left by the parts before).
//! Every decode step passes the per-step agree guard (all ranks present the same token, AGENTS §2.10).
//! RESULT: `DSV41_TP_MODEL_OK` when every gated point is within tolerance and every part ran.

use super::comm::{Dsv41Comm, TpComm};
use super::fwd::{HeadWant, TpModel};
use super::gate::TpGate;
use super::wload::RankSrc;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::dump::RefbDump;
use crate::dsv41::fwd::cmp::Pass;
use crate::dsv41::fwd::dev::Gpu;
use crate::tokenizer::QwenTokenizer;
use anyhow::{bail, Context, Result};
use cudarc::driver::CudaDevice;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn opt(name: &str) -> Option<String> { crate::opts::lookup(name).and_then(crate::opts::get).filter(|s| s != "unset") }

/// The program's options (all `Scope::Spmd`: they ride `TpConfig` to the nodes).
#[derive(Clone, Debug)]
pub struct ModelOpts { pub run: Vec<String>, pub fixtures: Vec<PathBuf>, pub prompt: String, pub gen_tokens: usize, pub chunk: usize, pub max_seq: usize, pub cache_mib: usize, pub lanes: usize,
                       /// the pack dir (the calibration cache's key); set by `run`
                       pub model_dir: PathBuf,
                       /** leg Q1: `--dsv41-q1-score-out` (the q1score part) */ pub q1_out: Option<PathBuf> }

impl ModelOpts {
    pub fn from_registry() -> Result<ModelOpts> { Self::from_registry_world(1) }
    pub fn from_registry_world(world: usize) -> Result<ModelOpts> {
        let run: Vec<String> = opt("dsv41-tp-run").unwrap_or_else(|| "gate,nll,gen,time".into()).split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        for r in &run { if !matches!(r.as_str(), "gate" | "nll" | "gen" | "time" | "q1score" | "lanes" | "spec" | "accept" | "bench" | "ced" | "graphs" | "width" | "sq1") {
            bail!("--dsv41-tp-run: unknown part {r:?} (gate, nll, gen, time, q1score, lanes, spec, accept, bench, ced, graphs, width, sq1)"); } }
        let fixtures = opt("dsv41-tp-fixtures").unwrap_or_else(|| "/mnt/models/dsv41/ref/short_chat/prefill_exact,/mnt/models/dsv41/ref/agent_4k/prefill_exact".into())
            .split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).map(PathBuf::from).collect();
        let prompt = opt("dsv41-tp-prompt").unwrap_or_else(|| DEFAULT_PROMPT.into());
        let num = |n: &str, d: usize| opt(n).and_then(|v| v.parse::<usize>().ok()).unwrap_or(d);
        let max_seq = opt("max-seq-len").and_then(|v| v.parse::<usize>().ok()).unwrap_or(16384);
        let def_cache = crate::dsv41::load::fit::default_engram_cache_mib(world) as usize;
        Ok(ModelOpts { run, fixtures, prompt, gen_tokens: num("dsv41-tp-gen-tokens", 1000), chunk: num("dsv41-tp-chunk", 256), max_seq,
                       cache_mib: num("engram-row-cache-mib", def_cache), lanes: num("dsv41-tp-lanes", 1).clamp(1, super::fwd::MAX_LANES),
                       model_dir: PathBuf::new(), q1_out: opt("dsv41-q1-score-out").map(PathBuf::from) })
    }
    pub fn has(&self, p: &str) -> bool { self.run.iter().any(|r| r == p) }
}

pub const DEFAULT_PROMPT: &str = "Write a detailed, well-structured essay of about 800 words on how the printing press changed science, \
    religion and politics in early modern Europe. Use concrete historical examples and finish with a short conclusion.";

/// The per-step agree guard (AGENTS §2.10).
pub struct Guard { step: u64, world: usize, pub n: u64 }
impl Guard {
    pub fn new(world: usize) -> Guard { Guard { step: 0, world, n: 0 } }
    pub fn agree(&mut self, what: &str, hash: u32) -> Result<()> {
        self.step += 1;
        if self.world < 2 { return Ok(()); }
        match crate::net::agree_ext(self.step, 0, 0, hash) {
            Some((_, w)) if w == crate::net::agree_word(0, hash) => { super::metrics::agreed(); self.n += 1; Ok(()) }
            Some((_, w)) => bail!("dsv41 TP agree MISMATCH at step {} ({what}): peer word {w:#x}, mine {:#x}", self.step, crate::net::agree_word(0, hash)),
            None => bail!("dsv41 TP agree: the link aborted or the peer did not arrive (step {}, {what})", self.step),
        }
    }
}

fn mem_info() -> (u64, u64) {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total); }
    (free as u64, total as u64)
}

fn rss_gib() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1048576.0)
}

/// Stop tokens of the pack (generation_config + config + tokenizer_config, AGENTS §3).
fn stop_ids(dir: &Path, tok: &QwenTokenizer) -> Vec<u32> {
    let mut v: Vec<u32> = Vec::new();
    for f in ["generation_config.json", "config.json"] {
        if let Ok(s) = std::fs::read_to_string(dir.join(f)) {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(&s) {
                for k in [j.get("eos_token_id"), j.get("text_config").and_then(|t| t.get("eos_token_id"))].into_iter().flatten() {
                    match k { serde_json::Value::Number(n) => v.extend(n.as_u64().map(|x| x as u32)), serde_json::Value::Array(a) => v.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32)), _ => {} }
                }
            }
        }
    }
    for t in ["<｜end▁of▁sentence｜>"] { if let Ok(ids) = tok.encode(t, false) { if ids.len() == 1 { v.push(ids[0]); } } }
    v.sort_unstable(); v.dedup();
    v
}

/// Run the program on this rank. `say` prints a line (every rank prints its own; the head's go to the report).
#[allow(clippy::too_many_arguments)]
pub fn run(dev: &Arc<CudaDevice>, comm: &TpComm, src: &RankSrc, model_dir: &Path, engram_dir: &Path, o: &ModelOpts, say: &mut dyn FnMut(String)) -> Result<bool> {
    let cfg: Dsv41Config = src.cfg.clone();
    let rank = comm.rank();
    crate::dsv41::gpu::load_module(dev)?;
    let cs = crate::gpu::fork_blocking_stream(dev);
    let g = Gpu { dev: dev.clone(), stream: cs.stream as u64 };
    let (free0, total) = mem_info();
    let (held, skipped) = src.held_bytes();
    let t0 = std::time::Instant::now();
    let mut m = TpModel::new(&g, &cs, &cfg, comm, src, engram_dir, model_dir, o.chunk.max(super::fwd::HEAD_MAX), o.max_seq, o.cache_mib, o.lanes, say)?;
    let want_spec = o.has("spec") || o.has("accept") || o.has("bench") || o.has("sq1");
    if opt("dsv41-tp-graphs").as_deref() != Some("off") {
        m.graphs = Some(super::graph::Graphs::default());
        say(format!("rank {rank}: decode / verify calls and DSpark draft steps run as CUDA graphs (captured per shape, replayed at every context)"));
    }
    if want_spec {
        let t = std::time::Instant::now();
        m.load_stages(src)?;
        say(format!("rank {rank}: DSpark stages 40-42 + main_proj loaded (split like the trunk) in {:.0} s; weights now {:.2} GiB", t.elapsed().as_secs_f64(),
                    m.weights_bytes as f64 / 1073741824.0));
    }
    let (free1, _) = mem_info();
    say(format!("rank {rank} memory: weights loaded {:.2} GiB (this rank's shard {:.2} GiB, of which {:.2} GiB DSpark / vision not loaded); drop in the driver's free memory {:.2} GiB (cuMemGetInfo, page cache included: read nvidia-smi for the process) \
                 (free {:.1} -> {:.1} GiB of {:.1}); process RSS {:.2} GiB; loaded in {:.0} s",
                m.weights_bytes as f64 / 1073741824.0, held as f64 / 1073741824.0, skipped as f64 / 1073741824.0, (free0 as f64 - free1 as f64) / 1073741824.0,
                free0 as f64 / 1073741824.0, free1 as f64 / 1073741824.0, total as f64 / 1073741824.0, rss_gib(), t0.elapsed().as_secs_f64()));
    say(format!("rank {rank} split: heads {}..{} ({} of {}), o_groups {}..{}, hc_fn K {}..{}, experts inner width (every expert), Engram heads {}..{}, vocabulary {}..{}; embedding {}",
                m.rg.h0, m.rg.h0 + m.rg.nh, m.rg.nh, cfg.heads, m.rg.g0, m.rg.g0 + m.rg.ng, m.rg.k0, m.rg.k0 + m.rg.klen, m.rg.eh0, m.rg.eh0 + m.rg.neh,
                comm.vocab().start, comm.vocab().end, if m.embed_bcast { "rank 0 gathers, the fp32 all-reduce broadcasts" } else { "held by every rank" }));
    say(comm.describe());
    // the first rendezvous after independent loads (the ranks' load times differ by minutes): the boot barrier's
    // deadline, not the per-step watchdog's 10 s
    m.barrier(0x4D4F)?;
    m.pa1_agree()?;   // INT12: the hc side stream only when every rank holds hc_fn whole (else off on every rank, one line)
    let mut guard = Guard { step: 0, world: comm.world(), n: 0 };
    guard.agree("model loaded", 0x4D4F_0001)?;
    let tok = QwenTokenizer::from_file(&model_dir.join("tokenizer.json").to_string_lossy()).context("tokenizer.json")?;
    let stops = stop_ids(model_dir, &tok);
    let mut ok = true;
    // ---------------------------------------------------------------- gate
    if o.has("gate") {
        m.barrier(0x5000 | 0)?;
        let mut gt = TpGate { rows: Vec::new(), pass: Pass::Exl3 };
        m.f.checks = true;
        for fx in &o.fixtures {
            let d = RefbDump::open(fx).with_context(|| format!("fixture {}", fx.display()))?;
            let name = fx_name(fx);
            say(format!("== TP gate on {name} (P {}) ==", d.prompt_len()?));
            let t = std::time::Instant::now();
            gt.run_fixture(&mut m, &d, &name, say)?;
            say(format!("   {name}: {:.0} s", t.elapsed().as_secs_f64()));
        }
        m.f.checks = false;
        let fails: Vec<String> = gt.rows.iter().filter(|r| r.pass == Some(false)).map(|r| format!("{} L{} {}", r.fixture, r.layer, r.point)).collect();
        let gated = gt.rows.iter().filter(|r| r.pass.is_some()).count();
        say(format!("TP gate (EXL3, world {}): {gated} gated points, {} within tolerance, {} not{}", comm.world(), gated - fails.len(), fails.len(),
                    if fails.is_empty() { String::new() } else { format!(": {}", fails.join("; ")) }));
        ok &= fails.is_empty() && gated > 0;
    }
    // ---------------------------------------------------------------- nll + the fixture's reply
    if o.has("nll") {
        m.barrier(0x5000 | 1)?;
        let fx = o.fixtures.iter().find(|f| RefbDump::open(f).ok().and_then(|d| d.prompt_len().ok()).map_or(false, |p| p >= 1000))
            .context("--dsv41-tp-run nll needs a fixture with P >= 1,000 in --dsv41-tp-fixtures")?;
        let d = RefbDump::open(fx)?;
        let ids: Vec<u32> = d.global("prompt_ids")?.i.iter().map(|&x| x as u32).collect();
        let p = ids.len();
        let lse = d.global("tf_lse")?.f;
        let tgt = d.global("tf_target_logit")?.f;
        m.reset_lane()?;
        let t = std::time::Instant::now();
        let (mut nll, mut nll_ref, mut n, mut top1, mut agree_ref) = (0f64, 0f64, 0usize, 0usize, 0usize);
        let ref_top: Vec<i64> = d.global("tf_top20_ids").map(|t| t.i.iter().step_by(20).copied().collect()).unwrap_or_default();
        let mut last = None;
        let mut r0 = 0;
        while r0 < p {
            let c = (p - r0).min(o.chunk);
            let probe: Vec<i32> = (0..c).map(|i| if r0 + i + 1 < p { ids[r0 + i + 1] as i32 } else { -1 }).collect();
            let outs = m.forward(&ids[r0..r0 + c], HeadWant::All, Some(&probe))?;
            for (i, o2) in outs.iter().enumerate() {
                let r = r0 + i;
                if r + 1 < p {
                    nll += -(o2.probe_lp as f64);
                    nll_ref += (lse[r] - tgt[r]) as f64;
                    n += 1;
                    if o2.greedy as u32 == ids[r + 1] { top1 += 1; }
                    if ref_top.get(r) == Some(&(o2.greedy as i64)) { agree_ref += 1; }
                }
            }
            last = outs.last().copied();
            guard.agree("nll chunk", outs.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
            r0 += c;
        }
        let secs = t.elapsed().as_secs_f64();
        say(format!("teacher-forced NLL on {} (P {p}, full non-CED prefill in {}-row chunks, {:.1} s = {:.0} tok/s): mean NLL {:.4} (REFb native {:.4}, same rows; delta {:+.4}), \
                     top-1 = next token {:.4} (REFb meta {:.4}), our greedy = REFb native's top-1 on {:.4} of rows",
                    fx.display(), o.chunk, secs, p as f64 / secs, nll / n as f64, nll_ref / n as f64, (nll - nll_ref) / n as f64, top1 as f64 / n as f64,
                    d.meta.get("teacher_forced_top1").and_then(|v| v.as_f64()).unwrap_or(f64::NAN), agree_ref as f64 / n as f64));
        let first = last.context("no head output")?.greedy as u32;
        let refdec: Vec<u32> = d.global("decode_tokens").map(|t| t.i.iter().map(|&x| x as u32).collect()).unwrap_or_default();
        let (text, toks, rate) = generate(&mut m, &mut guard, first, o.gen_tokens, &stops)?;
        let same = toks.iter().zip(refdec.iter()).take_while(|(a, b)| a == b).count();
        say(format!("greedy reply to {} ({} tokens, {:.1} tok/s decode at 1 row; the first {same} tokens equal REFb native's greedy continuation of {}):", fx.display(), toks.len(), rate, refdec.len()));
        say(format!("<<<{}>>>", tok.decode(&toks, false).unwrap_or(text)));
        m.settle()?;
    }
    // ---------------------------------------------------------------- q1score (leg Q1): per-row scores + logits taps
    if o.has("q1score") {
        let out_dir = o.q1_out.clone().context("--dsv41-tp-run q1score needs --dsv41-q1-score-out <dir>")?;
        std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
        for fx in &o.fixtures {
            m.barrier(0x5000 | 4)?;
            let d = RefbDump::open(fx).with_context(|| format!("fixture {}", fx.display()))?;
            let name = fx_name(fx);
            let mut ids: Vec<u32> = d.global("prompt_ids")?.i.iter().map(|&x| x as u32).collect();
            let p = ids.len();
            if let Ok(t) = d.global("decode_tokens") { ids.extend(t.i.iter().map(|&x| x as u32)); }
            let n = ids.len();
            // full logits: every row of a prompt up to 8,192 tokens; above, REFb's logit rows, the last 512 prompt rows
            // and the continuation
            let mut want: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            if n <= 8192 { want.extend(0..n); } else {
                if let Ok(r) = d.global("logits.rows") { want.extend(r.i.iter().map(|&x| x as usize)); }
                want.extend(p.saturating_sub(512)..n);
            }
            m.logit_tap = Some(super::q1tap::LogitTap::new(want));
            m.reset_lane()?;
            let t = std::time::Instant::now();
            let (mut greedy, mut lse, mut plp) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
            let mut r0 = 0;
            while r0 < n {
                let c = (n - r0).min(o.chunk);
                let probe: Vec<i32> = (0..c).map(|i| if r0 + i + 1 < n { ids[r0 + i + 1] as i32 } else { -1 }).collect();
                let outs = m.forward(&ids[r0..r0 + c], HeadWant::All, Some(&probe))?;
                for o2 in &outs { greedy.push(o2.greedy as i64); lse.push(o2.lse); plp.push(o2.probe_lp); }
                guard.agree("q1score chunk", outs.iter().fold(0u32, |a, x| a.wrapping_mul(31) ^ x.greedy as u32))?;
                r0 += c;
            }
            let secs = t.elapsed().as_secs_f64();
            let tap = m.logit_tap.take().context("logit tap")?;
            let v = comm.vocab();
            let path = out_dir.join(format!("{name}.rank{rank}.safetensors"));
            super::q1tap::write(&path, &ids, p, &greedy, &lse, &plp, &tap, (v.start, v.end))?;
            let nll_p: f64 = plp[..p - 1].iter().map(|x| -(*x as f64)).sum::<f64>() / (p - 1) as f64;
            let nll_c: f64 = if n > p { plp[p - 1..n - 1].iter().map(|x| -(*x as f64)).sum::<f64>() / (n - p) as f64 } else { f64::NAN };
            say(format!("q1score {name}: {p} prompt + {} continuation tokens in {secs:.1} s; mean NLL prompt rows {nll_p:.4}, continuation rows {nll_c:.4}; \
                         {} full-logit rows of vocabulary {}..{} -> {}", n - p, tap.rows.len(), v.start, v.end, path.display()));
        }
        m.settle()?;
    }
    // ---------------------------------------------------------------- gen
    if o.has("gen") {
        m.barrier(0x5000 | 2)?;
        use crate::dsv41::chat::{jx, render};
        let msgs = vec![jx::obj(vec![("role", jx::s("user")), ("content", jx::s(&o.prompt))])];
        let enc = render::encode_messages(&msgs, &render::EncodeOptions::new(render::ThinkingMode::Thinking)).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = tok.encode(&enc.prompt, false)?;
        m.reset_lane()?;
        let t = std::time::Instant::now();
        let mut last = None;
        let mut r0 = 0;
        while r0 < ids.len() {
            let c = (ids.len() - r0).min(o.chunk);
            let outs = m.forward(&ids[r0..r0 + c], if r0 + c == ids.len() { HeadWant::Last } else { HeadWant::None }, None)?;
            last = outs.last().copied().or(last);
            r0 += c;
        }
        let pre_s = t.elapsed().as_secs_f64();
        let first = last.context("no head output")?.greedy as u32;
        guard.agree("gen prefill", first)?;
        let (text, toks, rate) = generate(&mut m, &mut guard, first, o.gen_tokens, &stops)?;
        say(format!("greedy reply to the short prompt ({} prompt tokens, prefill {:.2} s; {} tokens generated at {:.1} tok/s, 1 row per step, no speculation, host launches):",
                    ids.len(), pre_s, toks.len(), rate));
        say(format!("<<<{}>>>", tok.decode(&toks, false).unwrap_or(text)));
        m.settle()?;
    }
    // ---------------------------------------------------------------- time
    if o.has("time") {
        m.barrier(0x5000 | 3)?;
        say(format!("decode step time (ms per call: the 40 trunk layers + the head exchange, host launches, no graph; context {} tokens):", m.lanes[0].pos));
        for n in [1usize, 6, 24] {
            let mut v = Vec::new();
            for k in 0..12 {
                let toks: Vec<u32> = (0..n).map(|i| 1000 + ((k * 7 + i) % 50000) as u32).collect();
                let t = std::time::Instant::now();
                let outs = m.forward(&toks, HeadWant::Last, None)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                guard.agree("time step", outs[0].greedy as u32)?;
                if k >= 2 { v.push(ms); }
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            m.settle()?;
            say(format!("  {n:>2} rows: median {:.2} ms, min {:.2}, max {:.2} (10 calls after 2 warm-ups)", v[v.len() / 2], v[0], v[v.len() - 1]));
        }
    }
    // ---------------------------------------------------------------- width (leg SE, GA's G4 probe)
    if o.has("width") {
        m.barrier(0x5000 | 5)?;
        ok &= super::width::run(&mut m, &mut guard, &tok, o.chunk, crate::dsv41::ced::schedule::mode_from_flags()?, say)?;
        m.settle()?;
    }
    // ---------------------------------------------------------------- the FW parts (lanes, spec, bench)
    if o.has("lanes") || o.has("ced") || o.has("graphs") || want_spec {
        let mut sp = if want_spec {
            let mut s = super::spec::TpSpec::new(&m, comm, &cs, m.lanes.len())?;
            say(s.attach_markov_cache(&mut m)?);   // leg PH1
            if let Some(l) = crate::dsv41::gpu::dspark::metrics::announce("dspark (TP, leg FW)", &format!("{:?}", s.policy.mode)) { say(l); }
            Some(s)
        } else { None };
        let mut o2 = o.clone();
        o2.model_dir = model_dir.to_path_buf();
        ok &= super::fwprog::run(&mut m, sp.as_mut(), &mut guard, &tok, &stops, &o2, say)?;
    }
    for l in crate::dsv41::fwd::receipts::boot_lines() { say(l); }
    if let Some(g) = &m.graphs { say(format!("rank {rank}: {}", g.line())); }
    say(format!("rank {rank}: agree guard passed {} times", guard.n));
    comm.check("end of the program")?;
    Ok(ok)
}

/// A fixture's name: its directory, or the parent's for a variant dir (`.../agent_4k/prefill_exact`).
fn fx_name(fx: &Path) -> String {
    let last = fx.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    if last.starts_with("prefill") { fx.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or(last) } else { last }
}

/// Greedy decode from `first` (already sampled) for up to `n` tokens or a stop token. Returns (text, tokens, tok/s).
fn generate(m: &mut TpModel, guard: &mut Guard, first: u32, n: usize, stops: &[u32]) -> Result<(String, Vec<u32>, f64)> {
    let mut toks = vec![first];
    let t = std::time::Instant::now();
    let mut cur = first;
    while toks.len() < n && !stops.contains(&cur) {
        let outs = m.forward(&[cur], HeadWant::Last, None)?;
        cur = outs[0].greedy as u32;
        guard.agree("decode step", cur ^ ((m.lanes[0].pos as u32) << 20))?;
        toks.push(cur);
    }
    let secs = t.elapsed().as_secs_f64();
    Ok((String::new(), toks.clone(), (toks.len().saturating_sub(1)) as f64 / secs.max(1e-9)))
}
