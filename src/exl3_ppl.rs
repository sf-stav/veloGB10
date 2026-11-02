//! S-B9-REL-PPL: `--perplexity` on an EXL3 pack — the SERVED chunked prefill with an all-rows
//! LM-head mode. A child module of `exl3_forward` (declared there with `#[path]`, like kvnll) so
//! the hook inside `prefill_chunk` can reach the private scratch buffers.
//!
//! WHAT IT MEASURES. Each window of `--window` tokens is prefilled from position 0 on a freshly
//! reset slot through `FwdModel::prefill_chunk` — the same function, kernels, chunk grid
//! (`--prefill-chunk`, default the served 4,095), QSA regime rule and KV format (`--kv-cache`) the
//! server runs. Where serving keeps only the LAST row of a chunk (`want_logits`), this mode
//! passes EVERY final-mixed row through the lm_head (`exl3_chain`, rows in groups bounded by the
//! chunk scratch), moves the f16 logits to the host and scores them in f64 (`kvnll::score_row`: the
//! max-shifted log-sum-exp the WP25 instrument already uses). Row i of a window predicts token
//! i+1; the window's first token is not scored (no BOS, raw text — the legacy `--perplexity`
//! convention), so a window of n tokens contributes n-1 NLL terms.
//!
//! REFERENCE. `--ppl-ref-windows N` re-runs the first N windows through the single-token decode
//! path (`forward_step`, m = 1: different kernels from the chunked prefill — GDN decode step,
//! decode attention, m=1 lm_head) and reports the per-token |dNLL|, so a wrong-but-consistent
//! prefill cannot hide behind its own gates (AGENTS §3).
//!
//! CHECKS THAT CAN FAIL (S-B9-REL-PPL2 W5, 1b). The first version's only guards were a width check
//! that compared the m <= 16 lm_head body with itself (a GEMM that writes nothing at both widths
//! yields the same stale rows and "0 differ") and a sanity bound the harness applied to its first
//! cell only; it accepted a PPL of 7.2e5 on the 4.05 pack. Now: (1) EVERY buffer the lm_head chain
//! writes (the final logits, the transformed input, the raw GEMM output, the split-K workspace) is
//! zeroed before every launch, and every group is scanned for rows that are all zero, mostly zero
//! or bit-identical to their neighbour; (2) each window of >= 64 scored tokens must clear
//! `--ppl-min-top1` and `--ppl-max-ppl` ON ITS OWN, and the running top-1 too, so a broken head
//! stops the run after its first window and one garbage window among good ones cannot hide in the
//! mean; (3) the final perplexity must lie inside `--ppl-min-ppl` .. `--ppl-max-ppl`. A violated
//! check is exit 1, NO `RESULT:` line, no per-token file under its real name. The decisions are
//! `head_chunk`, `RunJudge` and `finish_run`: functions the unit suite drives with a synthetic
//! device and synthetic windows. The defaults are the registry rows' (garbage guards for any
//! text); a harness that knows the text class passes its own tighter bounds
//! (`scripts/ppl_matrix.py`).

use super::kvnll::{f16_lut, read_ids, score_row, RowScore};
use super::*;

/// Rows of logits staged per lm_head group. 16 = the decode/verify contract width: the lm_head
/// chain runs the m <= 16 body (batch-invariant: row r of a group is bit-identical to that row run
/// alone at m = 1). That body decodes the 3-, 4- and 5-bit trellises and, in the K=6 module the
/// 4.05 pack's 6-bit lm_head needs, 6-bit; the loader itself accepts 1..=8 and every one of these
/// kernel switches ends in `default: break;`, a silent no-op, which is why the group scan below
/// looks for rows that were never written. The wide-M kernel (m > 16) instantiates only bits
/// 3/4/5: the first version of this tool sent the 6-bit lm_head through it with up to 101 rows
/// and scored stale buffer bytes (PPL 7e5, top-1 = chance).
pub(crate) const PPL_GROUP_ROWS: usize = 16;
/// "this row has no target inside the window" (the window's last row).
pub(crate) const NO_TARGET: u32 = u32::MAX;

// ---------------------------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------------------------

/// The run-level plausibility bounds (`--ppl-min-top1`, `--ppl-min-ppl`, `--ppl-max-ppl`). They are
/// GARBAGE guards, not quality gates: a healthy model on natural text sits far inside them, a head
/// that wrote nothing (top-1 at chance, a PPL beyond the vocabulary) far outside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunBounds {
    pub min_ppl: f64,
    pub max_ppl: f64,
    pub min_top1: f64,
}

impl RunBounds {
    /// The registry rows' defaults: ONE place states them (src/opts_table.rs).
    pub fn defaults() -> RunBounds {
        let d = |flag: &str| -> f64 {
            crate::opts::lookup(flag).and_then(|id| id.def().def.parse::<f64>().ok())
                .unwrap_or_else(|| panic!("--{flag} is not a registered numeric option"))
        };
        RunBounds { min_ppl: d("ppl-min-ppl"), max_ppl: d("ppl-max-ppl"), min_top1: d("ppl-min-top1") }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PplArgs {
    pub dir: String,
    /// raw text (tokenized here, no BOS) — or `ids`
    pub text: Option<String>,
    /// token ids (JSON array or whitespace) instead of text
    pub ids: Option<String>,
    pub window: usize,
    pub max_windows: usize,
    pub max_pos: usize,
    /// served prefill chunk width (rows per `prefill_chunk` call)
    pub chunk: usize,
    /// run the token-by-token decode reference on the first N windows (0 = off)
    pub ref_windows: usize,
    /// per-token CSV: window,pos,target,nll[,nll_decode]
    pub csv: Option<String>,
    /// write the scored windows' token ids (JSON) for an external runtime
    pub dump_ids: Option<String>,
    pub label: String,
    pub bounds: RunBounds,
}

/// Registered `--ppl-*` names (the "did you mean" candidates).
fn ppl_flag_names() -> Vec<&'static str> {
    crate::opts::REG.iter().map(|o| o.flag).filter(|f| f.starts_with("ppl-")).collect()
}

/// Every option of the tool, in registry order (the refusal's list).
fn tool_flag_names() -> Vec<&'static str> {
    crate::opts::REG.iter().filter(|o| o.cat == crate::opts::Cat::Ppl).map(|o| o.flag).collect()
}

/// Edit distance (the "did you mean" of an unknown `--ppl-` name).
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let sub = prev[j - 1] + (a[i - 1] != b[j - 1]) as usize;
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// `parse_cli` skips a name it does not know, so `--ppl-ref-windowz 3` used to drop the decode
/// reference in silence. In this tool's namespace an unknown name is an error.
pub(crate) fn refuse_unknown_ppl_names(args: &[String]) -> Result<()> {
    for a in args {
        let Some(body) = a.strip_prefix("--") else { continue };
        let name = body.split('=').next().unwrap_or("");
        if (name == "ppl" || name.starts_with("ppl-")) && crate::opts::lookup(name).is_none() {
            let near = ppl_flag_names().into_iter().map(|k| (edit_distance(name, k), k)).min();
            let hint = match near {
                Some((d, k)) if d <= 2 => format!(" Did you mean --{k}?"),
                _ => String::new(),
            };
            bail!("unknown option --{name} (the perplexity tool's options are --{}; it also reads --model-dir, \
                   --max-seq-len, --prefill-chunk and --kv-cache).{hint}", tool_flag_names().join(", --"));
        }
    }
    Ok(())
}

/// The KV format the tool measures: the four EXL3 formats only. `q4x` is the WP25 kvnll probe's
/// deliberately broken positive control (the server refuses it: a perplexity taken on it is wrong
/// by construction); the NVFP4 names would be written to options an EXL3 pack never reads, so the
/// run would measure the default q8 while the command line claimed another format.
pub(crate) fn check_kv_flag(v: Option<&str>) -> Result<()> {
    match v {
        None | Some("f32" | "f16" | "fp8" | "q8" | "q4k") => Ok(()),
        Some("q4x") => bail!("--kv-cache q4x is the WP25 kvnll probe's deliberately broken control; it is never \
                              served and a perplexity taken on it is wrong by construction (use f32, f16, fp8 or q8)"),
        Some(n @ ("bf16" | "q4" | "tq" | "tq3" | "k8v4" | "k8v8")) =>
            bail!("--kv-cache {n} is an NVFP4 KV format; an EXL3 pack takes f32, f16, fp8 or q8 (this flag would \
                   be ignored and the default q8 measured)"),
        Some(o) => bail!("--kv-cache must be f32, f16, fp8 or q8 for an EXL3 pack (got '{o}')"),
    }
}

/// The values `flag` is given on the command line, in order (`--flag v` and `--flag=v`).
fn values_of<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let (eq, bare) = (format!("--{flag}="), format!("--{flag}"));
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(v) = args[i].strip_prefix(&eq) {
            out.push(v);
        } else if args[i] == bare {
            if let Some(n) = args.get(i + 1) { out.push(n.as_str()); i += 1; }
        }
        i += 1;
    }
    out
}

/// A value-less Text / Path option takes the NEXT argument as its value, whatever it is: `--text
/// --mtp off` reads `--text = --mtp` and leaves `--mtp` at its default. That is the registry's
/// parse for every such option (a parser change for all options is its own reviewed item); in THIS
/// tool the next argument being another flag is refused, naming both. `--text=<v>` is not affected.
pub(crate) fn refuse_swallowed_flag(args: &[String]) -> Result<()> {
    const PATHY: [&str; 6] = ["text", "ppl-ids", "ppl-csv", "ppl-dump-ids", "ppl-label", "model-dir"];
    for (i, a) in args.iter().enumerate() {
        let Some(name) = a.strip_prefix("--") else { continue };
        if !PATHY.contains(&name) { continue; }
        if let Some(next) = args.get(i + 1) {
            if next.starts_with("--") {
                bail!("--{name} needs a value, but the next argument is {next}, which is another flag: a value-less \
                       --{name} would swallow it (every path or text option takes the next argument as its value). \
                       Give the value, or write --{name}=<value>");
            }
        }
    }
    Ok(())
}

/// A repeated option with DIFFERENT values: the registry options are last-wins and the hand-parsed
/// shared ones (`--model-dir`, `--kv-cache`, `--prefill-chunk`, `--max-seq-len`) are first-wins, so
/// `--kv-cache q4x --kv-cache q8` would be validated as q8 and measured as q4x. In this tool the
/// ambiguity is refused: pass each option once (the same value twice is fine).
pub(crate) fn refuse_conflicting_repeats(args: &[String]) -> Result<()> {
    let mut names = tool_flag_names();
    names.extend(["model-dir", "max-seq-len", "prefill-chunk", "kv-cache"]);
    for n in names {
        if n == "perplexity" { continue; }
        let vals = values_of(args, n);
        if let Some(first) = vals.first() {
            if let Some(other) = vals.iter().find(|v| *v != first) {
                bail!("--{n} is given twice with different values ('{first}' and '{other}'): the registry options \
                       are last-wins and --model-dir / --kv-cache / --prefill-chunk / --max-seq-len first-wins, so \
                       which one would be measured is ambiguous. Pass it once");
            }
        }
    }
    Ok(())
}

impl PplArgs {
    /// Legacy `--perplexity` defaults (`--window 1024 --max-windows 8 --max-seq-len 4096`) are kept
    /// so an old command line means the same thing; the chunk defaults to the SERVED value.
    ///
    /// The options are the registry's (src/opts_table.rs): `opts::parse_args` is the same walk,
    /// validation and last-wins rule as `parse_cli` without the store, so a value is typed and ranged
    /// exactly once. The refusals the registry cannot express live here.
    pub fn from_args(args: &[String]) -> Result<PplArgs> {
        refuse_unknown_ppl_names(args)?;
        refuse_swallowed_flag(args)?;
        refuse_conflicting_repeats(args)?;
        let vals = crate::opts::parse_args(args).map_err(|e| anyhow::anyhow!(e))?;
        let raw = |flag: &str| -> Option<String> {
            let id = crate::opts::lookup(flag).unwrap_or_else(|| panic!("--{flag} is not registered"));
            vals.iter().rev().find(|(i, _)| *i == id).and_then(|(_, v)| v.clone())
        };
        let int = |flag: &str| -> Result<usize> {
            let id = crate::opts::lookup(flag).unwrap_or_else(|| panic!("--{flag} is not registered"));
            crate::opts::ranged_from(id, raw(flag).as_deref()).map(|n| n as usize).map_err(|e| anyhow::anyhow!(e))
        };
        let float = |flag: &str, d: f64| -> Result<f64> {
            match raw(flag) {
                None => Ok(d),
                Some(s) => s.trim().parse::<f64>().ok().filter(|x| x.is_finite())
                    .ok_or_else(|| anyhow::anyhow!("--{flag} takes a finite number (got '{s}')")),
            }
        };
        let dir = crate::arg_value(args, "--model-dir").map(|s| s.to_string())
            .context("--perplexity on an EXL3 pack needs --model-dir <pack>")?;
        let text = raw("text");
        let ids = raw("ppl-ids");
        anyhow::ensure!(!(text.is_some() && ids.is_some()),
                        "--text and --ppl-ids are alternatives (text to tokenize, or token ids already); pass one");
        anyhow::ensure!(text.is_some() || ids.is_some(), "--perplexity needs --text <file> (or --ppl-ids <file>)");
        check_kv_flag(crate::arg_value(args, "--kv-cache"))?;
        let num = |f: &str, d: usize| -> Result<usize> {
            match crate::arg_value(args, f) {
                None => Ok(d),
                Some(s) => s.trim().parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("{f} must be a non-negative integer (got '{s}')")),
            }
        };
        let max_pos = match raw("max-seq-len") {
            None => 4096,
            Some(s) => s.trim().parse::<usize>()
                .map_err(|_| anyhow::anyhow!("--max-seq-len must be a non-negative integer (got '{s}')"))?,
        };
        let d = RunBounds::defaults();
        let bounds = RunBounds {
            min_ppl: float("ppl-min-ppl", d.min_ppl)?,
            max_ppl: float("ppl-max-ppl", d.max_ppl)?,
            min_top1: float("ppl-min-top1", d.min_top1)?,
        };
        anyhow::ensure!((0.0..=1.0).contains(&bounds.min_top1),
                        "--ppl-min-top1 is a fraction in 0..=1 (got {})", bounds.min_top1);
        anyhow::ensure!(bounds.min_ppl >= 1.0 && bounds.max_ppl > bounds.min_ppl,
                        "--ppl-min-ppl must be >= 1 (a perplexity below 1 is impossible) and below --ppl-max-ppl \
                         (got {} .. {})", bounds.min_ppl, bounds.max_ppl);
        let a = PplArgs {
            dir,
            text,
            ids,
            window: int("window")?,
            max_windows: int("max-windows")?,
            max_pos,
            chunk: num("--prefill-chunk", crate::exl3_serve::DEFAULT_PREFILL_CHUNK)?,
            ref_windows: int("ppl-ref-windows")?,
            csv: raw("ppl-csv"),
            dump_ids: raw("ppl-dump-ids"),
            label: raw("ppl-label").unwrap_or_else(|| "ppl".to_string()),
            bounds,
        };
        anyhow::ensure!(a.window >= 2, "--window must be >= 2 (one scored position needs two tokens)");
        anyhow::ensure!(a.chunk >= 1, "--prefill-chunk must be >= 1");
        anyhow::ensure!(a.window <= a.max_pos,
                        "--window {} exceeds --max-seq-len {} (raise --max-seq-len)", a.window, a.max_pos);
        Ok(a)
    }
}

// ---------------------------------------------------------------------------------------------
// pure helpers (unit-tested; no GPU)
// ---------------------------------------------------------------------------------------------

/// Consecutive non-overlapping windows `(start, len)` exactly as the legacy `toks.chunks(window)`
/// loop: stop at `max_windows`; a trailing chunk shorter than 2 tokens is not scored; a trailing
/// SHORT chunk (>= 2 tokens) is scored as a partial window.
pub(crate) fn plan_windows(ntok: usize, window: usize, max_windows: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = 0usize;
    while s < ntok && out.len() < max_windows {
        let len = window.min(ntok - s);
        if len < 2 { break; }
        out.push((s, len));
        s += len;
    }
    out
}

/// Targets for the rows `pos..end` of a window `win`: row i predicts `win[i + 1]`; the window's last
/// row has none.
pub(crate) fn row_targets(win: &[u32], pos: usize, end: usize) -> Vec<u32> {
    (pos..end).map(|i| if i + 1 < win.len() { win[i + 1] } else { NO_TARGET }).collect()
}

/// Rows per lm_head group: at most the decode contract width, at most what the chain scratch holds,
/// at most the chunk.
pub(crate) fn head_group_rows(yraw_len: usize, v: usize, c: usize) -> usize {
    (yraw_len / v.max(1)).min(PPL_GROUP_ROWS).min(c.max(1))
}

/// NLL of one logits row (f16 bits) against `target`, f64 log-sum-exp.
pub(crate) fn nll_row(row: &[u16], target: usize, lut: &[f32]) -> (f64, RowScore) {
    let rs = score_row(row, lut);
    let lt = lut[row[target] as usize] as f64;
    (rs.lse - lt, rs)
}

/// Running window / run totals.
#[derive(Debug, Default, Clone)]
pub(crate) struct PplAcc {
    pub sum: f64,
    pub n: usize,
    pub top1: usize,
    pub nonfinite_rows: usize,
    pub nonfinite_nll: usize,
}

impl PplAcc {
    pub fn add(&mut self, nll: f64, argmax: u32, target: u32, nonfinite_row: bool) {
        self.n += 1;
        self.top1 += (argmax == target) as usize;
        self.nonfinite_rows += nonfinite_row as usize;
        if nll.is_finite() { self.sum += nll; } else { self.nonfinite_nll += 1; }
    }
    pub fn merge(&mut self, o: &PplAcc) {
        self.sum += o.sum; self.n += o.n; self.top1 += o.top1;
        self.nonfinite_rows += o.nonfinite_rows; self.nonfinite_nll += o.nonfinite_nll;
    }
    pub fn mean_nll(&self) -> f64 { self.sum / self.n.max(1) as f64 }
    pub fn ppl(&self) -> f64 { self.mean_nll().exp() }
}

impl PplAcc {
    pub fn top1_frac(&self) -> f64 { self.top1 as f64 / self.n.max(1) as f64 }
}

/// What a group of `m` logits rows (`m * v` f16 values, staged on the host) shows about the lm_head
/// launch that filled it. Each count is a DEFECT: a healthy launch writes a distinct, essentially
/// zero-free row for every input row, and the device buffer was zeroed before the launch, so a
/// kernel that wrote nothing (the `default: break;` of its bit-width switch) leaves all-zero rows,
/// one that skipped part of its grid leaves rows with long runs of zeros, and one that broadcast or
/// kept stale output leaves bit-identical neighbours.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupDefects {
    /// rows whose every value is still the pre-launch zero
    pub unwritten: usize,
    /// rows that are not entirely zero but hold more than 1 zero in 20 (`SPARSE_ZERO_FRACTION`)
    pub sparse: usize,
    /// rows bit-identical to the row before them
    pub duplicate: usize,
}

/// A real row of f16 logits has next to no exact zeros: the most one could expect are the
/// vocabulary's padding columns (243 of 248,320 on the Qwen3.8 packs, 0.1%). More than 1 in 20
/// means part of the launch's grid never ran. (Not yet seen on a real pack: part 2 prints the
/// counts for both packs.)
pub(crate) const SPARSE_ZERO_FRACTION: usize = 20;

pub(crate) fn group_defects(host: &[u16], v: usize, m: usize) -> GroupDefects {
    let mut d = GroupDefects::default();
    for r in 0..m {
        let row = &host[r * v..(r + 1) * v];
        let zeros = row.iter().filter(|&&x| x == 0).count();
        if zeros == v { d.unwritten += 1; } else if zeros * SPARSE_ZERO_FRACTION > v { d.sparse += 1; }
        if r > 0 && row == &host[(r - 1) * v..r * v] { d.duplicate += 1; }
    }
    d
}

/// Scored tokens below which the top-1 floor is not applied AT ALL (per window, cumulative or at
/// the end): a handful of tokens says nothing about a model, and a valid model would fail the floor
/// by chance on a window of 2 to 10 tokens. A run that small says so on its `PPL_BOUNDS` line
/// (`NOT_APPLIED`); the perplexity bounds still apply to it.
pub(crate) const TOP1_MIN_SAMPLE: usize = 64;

/// The top-1 floor. `Err` = the run FAILS.
pub(crate) fn check_top1(n: usize, top1: f64, b: &RunBounds) -> std::result::Result<(), String> {
    if n >= TOP1_MIN_SAMPLE && !(top1 >= b.min_top1) {
        return Err(format!(
            "top-1 accuracy {top1:.4} over {n} scored tokens is below the floor {} (--ppl-min-top1). Chance for this \
             vocabulary is about 4e-6 and a healthy model on natural text scores 0.5 to 0.8: the logits are not \
             this model's (an lm_head that wrote nothing, a wrong KV format, a wrong window offset). The number is \
             NOT valid", b.min_top1));
    }
    Ok(())
}

/// The perplexity bound. `Err` = the run FAILS.
pub(crate) fn check_ppl(ppl: f64, b: &RunBounds) -> std::result::Result<(), String> {
    if !ppl.is_finite() || ppl < b.min_ppl || ppl > b.max_ppl {
        return Err(format!(
            "perplexity {ppl} is outside {} .. {} (--ppl-min-ppl / --ppl-max-ppl). Below the range the model was \
             told its own answers; above it the logits are not this model's. The number is NOT valid",
            b.min_ppl, b.max_ppl));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// the all-rows head (the hook prefill_chunk calls)
// ---------------------------------------------------------------------------------------------

/// Counters of the head scan, kept for the whole run and printed on the `PPL_HEAD_CHECK` line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeadCounters {
    /// lm_head launches (a liveness receipt: the all-rows path must have run)
    pub groups: usize,
    /// rows scanned by `group_defects`, and the defects found (the run stops at the first)
    pub rows_checked: usize,
    pub unwritten_rows: usize,
    pub sparse_rows: usize,
    pub duplicate_rows: usize,
    /// first-group width check: (f16 values differing, rows in that group) of group row 0 vs the same
    /// row run alone at m = 1 through the same chain (the batch-invariance contract). With one row in
    /// the group it compares m = 1 with m = 1 and says nothing about width: the report says so.
    pub width_check: Option<(usize, usize)>,
}

/// The buffers an lm_head chain launch writes (names only; `GpuHead` maps them to slices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum HeadBuf {
    /// the final f16 logits (`sink.lg`, or `psc.logits` for the m = 1 width check)
    Logits,
    /// the input transform's output (`exl3_had_suh` -> `chain_xh`)
    XhIn,
    /// the raw GEMM output (`chain_yraw`): what `exl3_hmma_gemm` writes and `exl3_had_svh` rewrites
    /// into the logits (the unfused path: the 6-bit lm_head of the 4.05 pack)
    YRaw,
    /// the split-K fp32 partials (`chain_ws`); only a chain with ks > 1
    KsWs,
}

/// The output buffer of each stage of an m <= 16 lm_head chain (`exl3_chain_impl`): the input
/// Hadamard always; then, for ks > 1, the split-K GEMM (partials), the ascending combine (raw) and
/// the output Hadamard (logits); for ks = 1 either the FUSED persistent GEMM (writes the logits
/// directly: 3/4/5 bits, when an `exl3_lmh_*` entry is usable) or the plain `exl3_hmma_gemm` (raw)
/// followed by the output Hadamard (logits). `fused` is a run-time property of the build and the
/// head width; the poison plan does not need it (it covers both).
pub(crate) fn chain_stage_outputs(fused: bool, ks: u32) -> Vec<HeadBuf> {
    let mut v = vec![HeadBuf::XhIn];
    if ks > 1 {
        v.extend([HeadBuf::KsWs, HeadBuf::YRaw, HeadBuf::Logits]);
    } else if fused {
        v.push(HeadBuf::Logits);
    } else {
        v.extend([HeadBuf::YRaw, HeadBuf::Logits]);
    }
    v
}

/// What to zero before a launch: every buffer ANY stage of the chain may write. A stage that
/// writes nothing then leaves zeros, which flow through the later (linear) stages into all-zero
/// logits rows, where the scan sees them. Zeroing only `Logits` observes the kernel under test only
/// where it writes the logits itself.
pub(crate) fn head_poison_plan(ks: u32) -> Vec<HeadBuf> {
    let mut v = vec![HeadBuf::Logits, HeadBuf::XhIn, HeadBuf::YRaw];
    if ks > 1 { v.push(HeadBuf::KsWs); }
    v
}

/// The device side of one lm_head group. The GPU implementation is `GpuHead`; the unit suite drives
/// the same host logic (`head_chunk`: poison -> launch -> read -> scan -> refuse -> score) with a
/// synthetic device, so "a launch that wrote nothing ends the run" is a tested function, not a
/// sequence of calls only a GPU run exercises (S-B9-REL-PPL2 1b item 2).
pub(crate) trait HeadDevice {
    /// Zero EVERY buffer the head chain writes for an m-row group: the final f16 logits and the
    /// intermediates (the Hadamard-transformed input, the raw GEMM output, the split-K workspace
    /// when the chain has one). Zeroing only the final buffer would leave the kernel under test
    /// unobserved on a head whose GEMM writes an intermediate that a later stage rewrites into the
    /// final one (the 6-bit lm_head of the 4.05 pack: exl3_hmma_gemm -> y_raw -> exl3_had_svh).
    fn poison(&mut self, m: usize) -> Result<()>;
    /// Stage hidden rows r0..r0+m, run the lm_head chain at m rows and copy the logits (>= m * v f16
    /// values, row-major) into `host`.
    fn run_group(&mut self, r0: usize, m: usize, host: &mut [u16]) -> Result<()>;
    /// The hidden row r0 ALONE at m = 1 through the same chain (buffers poisoned first); its v logits.
    fn run_single(&mut self, r0: usize, out: &mut [u16]) -> Result<()>;
    /// lm_head bits, for messages.
    fn bits(&self) -> i32;
}

/// Host-side state of one run's head: targets of the next chunk, the per-row NLL of the chunk just
/// run, the running accumulator, the staging buffer and the scan counters.
pub(crate) struct HeadState {
    /// targets of the rows of the NEXT chunk (set by the driver before each `prefill_chunk`)
    pub targets: Vec<u32>,
    /// per-row NLL of the chunk just run (NaN for unscored rows) — drained by the driver
    pub row_nll: Vec<f32>,
    pub acc: PplAcc,
    pub counters: HeadCounters,
    host: Vec<u16>,
    pub(crate) r_max: usize,
    h: usize,
    v: usize,
    lut: Vec<f32>,
    threads: usize,
}

impl HeadState {
    pub(crate) fn new(r_max: usize, h: usize, v: usize) -> HeadState {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 16);
        HeadState {
            targets: Vec::new(),
            row_nll: Vec::new(),
            acc: PplAcc::default(),
            counters: HeadCounters::default(),
            host: vec![0u16; r_max * v],
            r_max, h, v,
            lut: f16_lut(),
            threads,
        }
    }
}

/// The all-rows head of one prefill chunk of `c` rows: groups of up to `r_max` (<= 16) rows go
/// through the lm_head chain at the m <= 16 body — the batch-invariant decode/verify body, never the
/// wide-M kernel (it decodes 3/4/5 bits only) — into buffers that were ZEROED first; each group is
/// scanned for unwritten, mostly-zero and duplicated rows (any defect is an `Err`: the run ends) and
/// then scored on the host. `st.targets[i]` is row i's target (NO_TARGET = skip).
pub(crate) fn head_chunk<D: HeadDevice>(dev: &mut D, st: &mut HeadState, c: usize) -> Result<()> {
    let v = st.v;
    anyhow::ensure!(st.targets.len() == c,
                    "ppl: {} targets for a chunk of {c} rows (the driver must set them first)", st.targets.len());
    st.row_nll.clear();
    st.row_nll.resize(c, f32::NAN);
    let mut r0 = 0usize;
    while r0 < c {
        let m = st.r_max.min(c - r0);
        if st.targets[r0..r0 + m].iter().all(|&t| t == NO_TARGET) { r0 += m; continue; }
        // Poison first: a kernel whose bit-width switch falls to `default: break;` writes NOTHING,
        // and without this the buffers would still hold the previous group's data (a plausible-looking,
        // wrong number). Zero them, so "wrote nothing" is "all-zero rows", which the scan refuses.
        dev.poison(m)?;
        dev.run_group(r0, m, &mut st.host)?;
        st.counters.groups += 1;
        let defects = group_defects(&st.host[..m * v], v, m);
        st.counters.rows_checked += m;
        st.counters.unwritten_rows += defects.unwritten;
        st.counters.sparse_rows += defects.sparse;
        st.counters.duplicate_rows += defects.duplicate;
        anyhow::ensure!(defects == GroupDefects::default(),
            "ppl head check FAILED: of the {m} logits rows of group {} (chunk rows {r0}..{}), {} were never written \
             (all zero after the buffers were zeroed), {} are mostly zero (part of the launch never ran) and {} are \
             bit-identical to the row before them (lm_head bits {}). The logits are NOT valid",
            st.counters.groups, r0 + m, defects.unwritten, defects.sparse, defects.duplicate, dev.bits());
        // score the group on up to `threads` host threads (rows are independent)
        let tg: Vec<u32> = st.targets[r0..r0 + m].to_vec();
        let host = &st.host;
        let lut = &st.lut;
        let nt = st.threads.min(m).max(1);
        let per = (m + nt - 1) / nt;
        let outs: Vec<Vec<(f64, u32, bool)>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..nt).map(|t| {
                let (a, b) = (t * per, ((t + 1) * per).min(m));
                let tg = &tg;
                s.spawn(move || -> Vec<(f64, u32, bool)> {
                    (a..b).map(|r| {
                        if tg[r] == NO_TARGET { return (f64::NAN, u32::MAX, false); }
                        let (nll, rs) = nll_row(&host[r * v..(r + 1) * v], tg[r] as usize, lut);
                        (nll, rs.argmax, rs.nonfinite)
                    }).collect()
                })
            }).collect();
            hs.into_iter().map(|j| j.join().expect("ppl scoring thread")).collect()
        });
        let mut r = r0;
        for o in outs {
            for (nll, am, nf) in o {
                if tg[r - r0] != NO_TARGET {
                    st.acc.add(nll, am, tg[r - r0], nf);
                    st.row_nll[r] = nll as f32;
                }
                r += 1;
            }
        }
        // WIDTH CHECK (once per run, first scored group): group row 0 must equal the same hidden
        // row run ALONE at m = 1 through the same chain, f16 bit for f16 bit (the batch-invariance
        // contract). It is informative only when the group has >= 2 rows, and it cannot see a
        // kernel that writes nothing at both widths: the poison and the scan above do that.
        if st.counters.width_check.is_none() {
            let mut one = vec![0u16; v];
            dev.run_single(r0, &mut one)?;
            let differ = one.iter().zip(&st.host[..v]).filter(|(a, b)| a != b).count();
            st.counters.width_check = Some((differ, m));
            anyhow::ensure!(differ == 0,
                "ppl head width check FAILED: row {r0} run as part of a {m}-row group differs from the same row \
                 at m = 1 in {differ} of {v} f16 logits (lm_head bits {}); the logits are NOT valid", dev.bits());
        }
        r0 += m;
    }
    Ok(())
}

/// Staging + results of one `--perplexity` run; lives in `PrefillScratch::ppl` (None = serving).
pub(crate) struct PplSink {
    pub st: HeadState,
    xr: CudaSlice<u16>,
    lg: CudaSlice<u16>,
}

impl PplSink {
    pub(crate) fn new(model: &FwdModel, psc: &PrefillScratch) -> Result<PplSink> {
        let (h, v) = (model.cfg.hidden_size, model.cfg.vocab_size);
        anyhow::ensure!(model.tp.is_none(), "--perplexity is a single-process (TP=1) tool");
        let r_max = head_group_rows(psc.chain_yraw.len(), v, psc.c);
        anyhow::ensure!(r_max >= 1, "prefill scratch chain_yraw ({}) cannot hold one logits row of {v}",
                        psc.chain_yraw.len());
        Ok(PplSink {
            st: HeadState::new(r_max, h, v),
            xr: model.dev.alloc_zeros::<u16>(r_max * h)?,
            lg: model.dev.alloc_zeros::<u16>(r_max * v)?,
        })
    }
}

/// The GPU implementation of `HeadDevice`: the lm_head `Quad` through `exl3_chain`.
struct GpuHead<'a, 'l> {
    model: &'a FwdModel,
    l: &'a Launcher<'l>,
    psc: &'a mut PrefillScratch,
    xr: &'a mut CudaSlice<u16>,
    lg: &'a mut CudaSlice<u16>,
    h: usize,
}

impl GpuHead<'_, '_> {
    /// Execute `head_poison_plan` for `m` rows. `logits` is the final output buffer of THIS launch
    /// (`sink.lg` for a group, `psc.logits` for the m = 1 width check).
    fn poison_buffers(&mut self, m: usize, group_logits: bool) -> Result<()> {
        let (k, n, ks) = (self.model.lm_head.k as usize, self.model.lm_head.n as usize, self.model.lm_head.ks);
        let dev = &self.model.dev;
        for b in head_poison_plan(ks) {
            match b {
                HeadBuf::Logits if group_logits => dev.memset_zeros(&mut *self.lg)?,
                HeadBuf::Logits => dev.memset_zeros(&mut self.psc.logits)?,
                HeadBuf::XhIn => dev.memset_zeros(&mut self.psc.chain_xh.slice_mut(0..m * k))?,
                HeadBuf::YRaw => dev.memset_zeros(&mut self.psc.chain_yraw.slice_mut(0..m * n))?,
                HeadBuf::KsWs => {
                    let len = (ks as usize * m * n).min(self.psc.chain_ws.len());
                    dev.memset_zeros(&mut self.psc.chain_ws.slice_mut(0..len))?
                }
            }
        }
        Ok(())
    }
}

impl HeadDevice for GpuHead<'_, '_> {
    fn poison(&mut self, m: usize) -> Result<()> {
        self.poison_buffers(m, true)
    }

    fn run_group(&mut self, r0: usize, m: usize, host: &mut [u16]) -> Result<()> {
        let h = self.h;
        // stage rows r0..r0+m of psc.x (the compute stream orders this after the final mix)
        xqlaunch!(self.l, "xq_copy_row", ((((m * h) as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                  (&mut *self.xr, &self.psc.x, (r0 * h) as i64, (m * h) as i32))?;
        exl3_chain(self.l, &self.model.lm_head, &*self.xr, &mut self.psc.chain_xh, &mut self.psc.chain_yraw,
                   &mut *self.lg, m, &self.psc.chain_ws)?;
        self.model.dev.synchronize()?;
        self.model.dev.dtoh_sync_copy_into(&*self.lg, host)?;
        Ok(())
    }

    fn run_single(&mut self, r0: usize, out: &mut [u16]) -> Result<()> {
        let h = self.h;
        xqlaunch!(self.l, "xq_copy_row", ((((h) as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                  (&mut *self.xr, &self.psc.x, (r0 * h) as i64, h as i32))?;
        self.poison_buffers(1, false)?;
        exl3_chain(self.l, &self.model.lm_head, &*self.xr, &mut self.psc.chain_xh, &mut self.psc.chain_yraw,
                   &mut self.psc.logits, 1, &self.psc.chain_ws)?;
        self.model.dev.synchronize()?;
        let one = self.model.dev.dtoh_sync_copy(&self.psc.logits)?;
        anyhow::ensure!(one.len() >= out.len(), "psc.logits holds {} values, one logits row is {}", one.len(), out.len());
        out.copy_from_slice(&one[..out.len()]);
        Ok(())
    }

    fn bits(&self) -> i32 { self.model.lm_head.bits }
}

impl FwdModel {
    /// The all-rows head of one prefill chunk (the hook `prefill_chunk` calls): `psc.x` holds the
    /// chunk's `c` final-mixed rows (f16). The host logic is `head_chunk`.
    pub(super) fn ppl_all_rows(&self, l: &Launcher, psc: &mut PrefillScratch, sink: &mut PplSink,
                               c: usize) -> Result<()> {
        let PplSink { st, xr, lg } = sink;
        let h = st.h;
        let mut dev = GpuHead { model: self, l, psc, xr, lg, h };
        head_chunk(&mut dev, st, c)
    }
}

// ---------------------------------------------------------------------------------------------
// the run: windows, judgement, conclusion
// ---------------------------------------------------------------------------------------------

/// What one scored window yields.
pub(crate) struct WindowScore {
    pub acc: PplAcc,
    /// per-row NLL (NaN where unscored)
    pub nll: Vec<f32>,
    pub secs: f64,
}

/// Where windows come from: the GPU (`GpuWindows`) or, in the unit suite, a synthetic source.
pub(crate) trait WindowSource {
    fn score_window(&mut self, wi: usize, win: &[u32]) -> Result<WindowScore>;
    fn counters(&self) -> HeadCounters;
}

/// The run's judgement, fed one window at a time and then asked to conclude. Every decision "this
/// run is invalid" is here, so the unit suite can drive it with synthetic windows.
pub(crate) struct RunJudge {
    pub bounds: RunBounds,
    pub total: PplAcc,
    pub windows: usize,
}

/// What a passed run reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RunSummary {
    pub tokens: usize,
    pub mean_nll: f64,
    pub ppl: f64,
    pub top1: f64,
    /// false = the run scored fewer than `TOP1_MIN_SAMPLE` tokens, so no top-1 floor applied
    pub top1_floor_applied: bool,
}

impl RunJudge {
    pub(crate) fn new(bounds: RunBounds) -> RunJudge {
        RunJudge { bounds, total: PplAcc::default(), windows: 0 }
    }

    /// Judge window `wi` (1-based) as it lands. `Err` = the run FAILS now. A window of at least
    /// `TOP1_MIN_SAMPLE` scored tokens must clear the bounds ON ITS OWN: one garbage window among
    /// good ones moves the cumulative top-1 and perplexity too little to be seen.
    pub(crate) fn after_window(&mut self, wi: usize, expected: usize, w: &PplAcc) -> std::result::Result<(), String> {
        if w.n != expected {
            return Err(format!("window {wi}: {} positions scored, expected {expected} (the all-rows head did not \
                                run for every row)", w.n));
        }
        if w.nonfinite_rows != 0 || w.nonfinite_nll != 0 {
            return Err(format!("window {wi}: non-finite logits/NLL in {} rows / {} terms - the number is NOT valid",
                               w.nonfinite_rows, w.nonfinite_nll));
        }
        check_top1(w.n, w.top1_frac(), &self.bounds).map_err(|e| format!("window {wi} (alone): {e}"))?;
        if w.n >= TOP1_MIN_SAMPLE && w.ppl() > self.bounds.max_ppl {
            return Err(format!("window {wi} (alone): perplexity {} is above --ppl-max-ppl {}: the logits of this \
                                window are not this model's. The number is NOT valid", w.ppl(), self.bounds.max_ppl));
        }
        self.total.merge(w);
        self.windows += 1;
        // fail fast on the running total too (windows too short for the per-window reading)
        check_top1(self.total.n, self.total.top1_frac(), &self.bounds).map_err(|e| format!("after window {wi}: {e}"))
    }

    /// Conclude: the bounds on the whole run. `Err` = no `RESULT:` line.
    pub(crate) fn conclude(&self, head: &HeadCounters) -> std::result::Result<RunSummary, String> {
        if self.total.n == 0 || head.groups == 0 {
            return Err(format!("no token scored ({} groups): the all-rows lm_head path never ran", head.groups));
        }
        if self.total.nonfinite_rows != 0 || self.total.nonfinite_nll != 0 {
            return Err(format!("non-finite logits/NLL in {} rows / {} terms - the number is NOT valid",
                               self.total.nonfinite_rows, self.total.nonfinite_nll));
        }
        let mean = self.total.mean_nll();
        if !(mean.is_finite() && mean > 0.0) {
            return Err(format!("mean NLL {mean} is not a plausible value"));
        }
        check_top1(self.total.n, self.total.top1_frac(), &self.bounds)?;
        check_ppl(mean.exp(), &self.bounds)?;
        Ok(RunSummary { tokens: self.total.n, mean_nll: mean, ppl: mean.exp(), top1: self.total.top1_frac(),
                        top1_floor_applied: self.total.n >= TOP1_MIN_SAMPLE })
    }
}

/// Score every window through `src`, printing one line each to `out` and judging each as it lands.
/// Stops at the first window that fails: returns the per-window NLLs scored so far and the reason.
pub(crate) fn score_all<S: WindowSource>(src: &mut S, wins: &[(usize, usize)], ids: &[u32], judge: &mut RunJudge,
                                         out: &mut dyn std::io::Write)
                                         -> Result<(Vec<Vec<f32>>, Option<String>)> {
    let mut per_window: Vec<Vec<f32>> = Vec::new();
    for (wi, &(s, n)) in wins.iter().enumerate() {
        let ws = src.score_window(wi, &ids[s..s + n])?;
        writeln!(out, "  window {:2}: {:5} tok   nll/tok {:.4}   ppl {:.3}   top1 {:.3}   nonfinite {}   {:.1}s",
                 wi + 1, ws.acc.n, ws.acc.mean_nll(), ws.acc.ppl(), ws.acc.top1_frac(),
                 ws.acc.nonfinite_rows + ws.acc.nonfinite_nll, ws.secs)?;
        per_window.push(ws.nll);
        if let Err(e) = judge.after_window(wi + 1, n - 1, &ws.acc) {
            return Ok((per_window, Some(e)));
        }
    }
    Ok((per_window, None))
}

/// Everything after the device work: print the head / stats / bounds lines, decide, write the
/// per-token files, print `RESULT:`. A failed run (a window verdict in `failure`, or a failed
/// conclusion) prints NO `RESULT:` line, writes its per-token CSV only as `<csv>.INVALID` with the
/// reason on its first line, writes no ids dump, and returns the reason as `Err`.
pub(crate) struct FinishInputs<'a> {
    pub a: &'a PplArgs,
    pub kv_name: &'a str,
    pub wins: &'a [(usize, usize)],
    pub ids: &'a [u32],
    pub vocab: usize,
    pub src: &'a str,
    pub per_window_nll: &'a [Vec<f32>],
    pub dec_nll: &'a [Vec<f32>],
    pub head: HeadCounters,
    pub elapsed_s: f64,
}

pub(crate) fn finish_run(fi: &FinishInputs, judge: &RunJudge, failure: Option<String>,
                         out: &mut dyn std::io::Write) -> Result<()> {
    let a = fi.a;
    let verdict: std::result::Result<RunSummary, String> = match failure {
        Some(e) => Err(e),
        None => judge.conclude(&fi.head),
    };
    writeln!(out)?;
    let width = match fi.head.width_check {
        Some((d, rows)) if rows >= 2 => format!("{d}_of_1_rows_differ_vs_m1(group_rows={rows})"),
        Some((_, rows)) => format!("not_informative(group_rows={rows})"),
        None => "not_run".to_string(),
    };
    writeln!(out, "PPL_HEAD_CHECK groups={} rows_scanned={} unwritten_rows={} sparse_rows={} duplicate_rows={} width_check={}",
             fi.head.groups, fi.head.rows_checked, fi.head.unwritten_rows, fi.head.sparse_rows,
             fi.head.duplicate_rows, width)?;
    writeln!(out, "PPL_STATS kv={} windows={} lm_head_groups={} top1={:.4} elapsed={:.1}s",
             fi.kv_name, judge.windows, fi.head.groups, judge.total.top1_frac(), fi.elapsed_s)?;
    let floor = if judge.total.n >= TOP1_MIN_SAMPLE { "applied".to_string() }
                else { format!("NOT_APPLIED(only {} scored tokens < {TOP1_MIN_SAMPLE})", judge.total.n) };
    writeln!(out, "PPL_BOUNDS ppl {} .. {}  top1 >= {} ({floor}; each window of >= {TOP1_MIN_SAMPLE} tokens is judged alone too)",
             a.bounds.min_ppl, a.bounds.max_ppl, a.bounds.min_top1)?;

    // per-token outputs: only a PASSED run owns its file names
    let invalid = verdict.as_ref().err().map(|e| e.clone());
    if let Some(p) = &a.csv {
        let target = if invalid.is_some() { format!("{p}.INVALID") } else { p.clone() };
        let mut w = std::io::BufWriter::new(std::fs::File::create(&target).with_context(|| format!("create {target}"))?);
        if let Some(e) = &invalid { writeln!(w, "# INVALID RUN - {e}")?; }
        writeln!(w, "label,window,pos,target,nll,nll_decode")?;
        for (wi, nl) in fi.per_window_nll.iter().enumerate() {
            let (s, n) = fi.wins[wi];
            for i in 0..n - 1 {
                let dd = fi.dec_nll.get(wi).and_then(|d| d.get(i)).map(|x| format!("{x:.6}")).unwrap_or_default();
                writeln!(w, "{},{},{},{},{:.6},{}", a.label, wi + 1, i, fi.ids[s + i + 1], nl[i], dd)?;
            }
        }
        w.flush()?;
    }
    if invalid.is_none() {
        if let Some(p) = &a.dump_ids {
            let ws: Vec<&[u32]> = fi.wins.iter().map(|&(s, n)| &fi.ids[s..s + n]).collect();
            std::fs::write(p, serde_json::to_string(&serde_json::json!({
                "windows": ws, "vocab": fi.vocab, "source": fi.src, "no_bos": true,
            }))?).with_context(|| format!("write {p}"))?;
        }
    }
    match verdict {
        Ok(s) => {
            writeln!(out, "RESULT: tokens={}  mean_nll={:.5}  PERPLEXITY={:.4}", s.tokens, s.mean_nll, s.ppl)?;
            Ok(())
        }
        Err(e) => bail!("{e}"),
    }
}

// ---------------------------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------------------------

fn fnv64(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in ids { for b in t.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); } }
    h
}

/// Windows from the served chunked prefill on the GPU.
struct GpuWindows<'a> {
    model: &'a FwdModel,
    psc: &'a mut PrefillScratch,
    chunk: usize,
}

impl WindowSource for GpuWindows<'_> {
    fn score_window(&mut self, _wi: usize, win: &[u32]) -> Result<WindowScore> {
        let n = win.len();
        let tw = std::time::Instant::now();
        self.model.reset_slot(0)?;
        let mut nll_w: Vec<f32> = Vec::with_capacity(n);
        let before = self.psc.ppl.as_ref().unwrap().st.acc.clone();
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + self.chunk).min(n);
            let toks: Vec<i32> = win[pos..end].iter().map(|&t| t as i32).collect();
            self.psc.ppl.as_mut().unwrap().st.targets = row_targets(win, pos, end);
            self.model.prefill_chunk(self.psc, &toks, pos, 0, false, None)?;
            nll_w.extend(self.psc.ppl.as_ref().unwrap().st.row_nll.iter().copied());
            pos = end;
        }
        self.model.dev.synchronize()?;
        let mut wacc = self.psc.ppl.as_ref().unwrap().st.acc.clone();
        // window totals = run totals - totals before this window
        wacc.sum -= before.sum; wacc.n -= before.n; wacc.top1 -= before.top1;
        wacc.nonfinite_rows -= before.nonfinite_rows; wacc.nonfinite_nll -= before.nonfinite_nll;
        Ok(WindowScore { acc: wacc, nll: nll_w, secs: tw.elapsed().as_secs_f64() })
    }

    fn counters(&self) -> HeadCounters { self.psc.ppl.as_ref().unwrap().st.counters }
}

/// `--perplexity --model-dir <EXL3 pack> --text <file> [--window N --max-windows N --max-seq-len N
/// --prefill-chunk C --kv-cache q8|fp8|f16|f32 --ppl-ref-windows N --ppl-csv f --ppl-dump-ids f
/// --ppl-ids f --ppl-label s --ppl-min-top1 X --ppl-min-ppl X --ppl-max-ppl X]` (all registered:
/// src/opts_table.rs).
pub fn perplexity_exl3(a: &PplArgs) -> Result<()> {
    let t0 = std::time::Instant::now();
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir.trim_end_matches('/')))?;
    let (ids, src): (Vec<u32>, String) = match (&a.ids, &a.text) {
        (Some(p), _) => (read_ids(p)?, p.clone()),
        (None, Some(p)) => {
            let txt = std::fs::read_to_string(p).with_context(|| format!("read {p}"))?;
            // No chat template, no BOS: score raw text (the legacy --perplexity convention).
            (tok.encode(&txt, false)?, p.clone())
        }
        (None, None) => bail!("--perplexity needs --text or --ppl-ids"),
    };
    println!("Perplexity: {} tokens from {}, window={}, max_windows={}", ids.len(), src, a.window, a.max_windows);
    let wins = plan_windows(ids.len(), a.window, a.max_windows);
    anyhow::ensure!(!wins.is_empty(), "no scoreable window: {} tokens, window {}", ids.len(), a.window);
    // The run OWNS its output names from here: a file left under them by an earlier run is cleared
    // (the first version truncated the CSV at this point too), and only a PASSED run creates them
    // again. A path that cannot be written is found NOW, not after the run: probe it.
    for p in [&a.csv, &a.dump_ids].into_iter().flatten() {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(format!("{p}.INVALID"));
        let probe = format!("{p}.partial");
        std::fs::File::create(&probe).with_context(|| format!("cannot write {p} (create {probe})"))?;
        let _ = std::fs::remove_file(&probe);
    }

    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    let v = model.cfg.vocab_size;
    anyhow::ensure!(ids.iter().all(|&t| (t as usize) < v), "corpus has token ids >= vocab {v}");
    let fmt = model.kv_fmt;
    let chunk = a.chunk.min(a.window);
    let mut psc = model.prefill_scratch(chunk)?;
    let sink = PplSink::new(&model, &psc)?;
    println!("PPL_IDENT label={} kv={} chunk={} max_pos={} qsa_live_at_window_end={} windows={} \
              tokens_scored={} ids_fnv={:016x} head_group_rows={} lm_head_bits={} mtp_head={}",
             a.label, kv_fmt_name(fmt), chunk, model.max_pos(), model.qsa_live(a.window - 1), wins.len(),
             wins.iter().map(|w| w.1 - 1).sum::<usize>(), fnv64(&ids[..wins.last().map(|w| w.0 + w.1).unwrap()]),
             sink.st.r_max, model.lm_head.bits, model.mtp.is_some());
    psc.ppl = Some(sink);

    let mut judge = RunJudge::new(a.bounds);
    let (per_window_nll, failure, head) = {
        let mut gw = GpuWindows { model: &model, psc: &mut psc, chunk };
        let (nll, failure) = score_all(&mut gw, &wins, &ids, &mut judge, &mut std::io::stdout())?;
        (nll, failure, gw.counters())
    };

    // ---- reference: the same windows through the single-token decode path (a passed run only) ----
    let mut dec_nll: Vec<Vec<f32>> = Vec::new();
    let nref = if failure.is_some() { 0 } else { a.ref_windows.min(wins.len()) };
    if nref > 0 {
        let mut sc = Scratch::new(&model.dev, &model.cfg, 1, model.cfg.rotary_dim, model.max_pos())?;
        let lut = f16_lut();
        let (mut sum_abs, mut max_abs, mut cnt, mut agree, mut am_vs_step) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize);
        let (mut sum_p, mut sum_d) = (0.0f64, 0.0f64);
        for wi in 0..nref {
            let (s, n) = wins[wi];
            let win = &ids[s..s + n];
            let td = std::time::Instant::now();
            model.reset_slot(0)?;
            let mut d: Vec<f32> = Vec::with_capacity(n);
            let (mut wsum_p, mut wsum_d, mut wmax) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..n - 1 {
                let am = model.forward_step(&mut sc, &[win[i] as i32], &[i], &[0], None)?;
                let lg = model.logits_host(&sc)?;
                let (nll, rs) = nll_row(&lg[..v], win[i + 1] as usize, &lut);
                am_vs_step += (am.first().copied().unwrap_or(-1) as i64 != rs.argmax as i64) as usize;
                let p = per_window_nll[wi][i] as f64;
                let diff = (p - nll).abs();
                sum_abs += diff; max_abs = max_abs.max(diff); cnt += 1;
                wmax = wmax.max(diff);
                wsum_p += p; wsum_d += nll;
                // top-1 agreement of the decode path with the prefill path is judged on the NLL, not argmax
                agree += (diff < 0.05) as usize;
                d.push(nll as f32);
            }
            sum_p += wsum_p; sum_d += wsum_d;
            println!("  decode-ref window {:2}: {:5} tok   nll/tok prefill {:.5}  decode {:.5}   d(mean) {:+.5}   max|d| {:.4}   ppl {:.3} vs {:.3}   {:.1}s",
                     wi + 1, n - 1, wsum_p / (n - 1) as f64, wsum_d / (n - 1) as f64,
                     (wsum_p - wsum_d) / (n - 1) as f64, wmax,
                     (wsum_p / (n - 1) as f64).exp(), (wsum_d / (n - 1) as f64).exp(), td.elapsed().as_secs_f64());
            d.push(f32::NAN);
            dec_nll.push(d);
        }
        println!("PPL_REF windows={nref} tokens={cnt} mean|dNLL|={:.6} max|dNLL|={:.4} d(mean NLL)={:+.6} \
                  ppl_prefill={:.4} ppl_decode={:.4} frac(|d|<0.05)={:.4} step_argmax_vs_row_argmax_mismatch={}",
                 sum_abs / cnt as f64, max_abs, (sum_p - sum_d) / cnt as f64,
                 (sum_p / cnt as f64).exp(), (sum_d / cnt as f64).exp(), agree as f64 / cnt as f64, am_vs_step);
    }

    let fi = FinishInputs {
        a, kv_name: kv_fmt_name(fmt), wins: &wins, ids: &ids, vocab: v, src: &src,
        per_window_nll: &per_window_nll, dec_nll: &dec_nll, head, elapsed_s: t0.elapsed().as_secs_f64(),
    };
    finish_run(&fi, &judge, failure, &mut std::io::stdout())
}

// ---------------------------------------------------------------------------------------------
// tests (no GPU)
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn h(vals: &[f32]) -> Vec<u16> { vals.iter().map(|&x| f16::from_f32(x).to_bits()).collect() }

    #[test]
    fn nll_of_a_hand_computed_row() {
        // logits [1, 2, 0, -1], target 1 -> ln(e + e^2 + 1 + e^-1) - 2 = 0.4401896985611953
        let lut = f16_lut();
        let (nll, rs) = nll_row(&h(&[1.0, 2.0, 0.0, -1.0]), 1, &lut);
        assert!((nll - 0.4401896985611953).abs() < 1e-9, "{nll}");
        assert_eq!(rs.argmax, 1);
        assert!(!rs.nonfinite);
        // uniform row, any target -> ln 4
        let (nll, _) = nll_row(&h(&[0.0; 4]), 3, &lut);
        assert!((nll - 1.3862943611198906).abs() < 1e-12, "{nll}");
        // large logits: the max-shift must keep it finite and exact
        let (nll, _) = nll_row(&h(&[100.0, 99.0, -100.0, 100.0]), 1, &lut);
        assert!((nll - 1.8619948040582557).abs() < 1e-9, "{nll}");
    }

    #[test]
    fn accumulation_matches_the_hand_computed_perplexity() {
        let lut = f16_lut();
        let rows = [(h(&[1.0, 2.0, 0.0, -1.0]), 1usize), (h(&[0.0; 4]), 0usize)];
        let mut acc = PplAcc::default();
        for (row, t) in &rows {
            let (nll, rs) = nll_row(row, *t, &lut);
            acc.add(nll, rs.argmax, *t as u32, rs.nonfinite);
        }
        assert_eq!(acc.n, 2);
        // row 0: argmax = index 1 = target; row 1: a 4-way tie resolves to the lowest index (0) = target
        assert_eq!(acc.top1, 2);
        assert!((acc.mean_nll() - 0.9132420298405429).abs() < 1e-9);
        assert!((acc.ppl() - 2.4923898513482348).abs() < 1e-8);
    }

    #[test]
    fn head_groups_never_exceed_the_decode_width_or_the_scratch() {
        let v = 248_320;
        assert_eq!(head_group_rows(2048 * 12288, v, 2048), 16);          // the served chunk: capped at 16
        assert_eq!(head_group_rows(4095 * 12288, v, 4095), 16);
        assert_eq!(head_group_rows(3 * v + 5, v, 2048), 3);               // scratch holds only 3 rows
        assert_eq!(head_group_rows(v - 1, v, 2048), 0);                   // cannot hold one row (refused upstream)
        assert_eq!(head_group_rows(2048 * 12288, v, 5), 5);               // a 5-row chunk
        assert!(PPL_GROUP_ROWS <= 16, "above 16 rows the lm_head leaves the m<=16 body");
    }

    #[test]
    fn nonfinite_terms_are_counted_not_summed() {
        let mut acc = PplAcc::default();
        acc.add(1.0, 0, 0, false);
        acc.add(f64::NAN, 0, 1, true);
        assert_eq!(acc.n, 2);
        assert_eq!(acc.nonfinite_nll, 1);
        assert_eq!(acc.nonfinite_rows, 1);
        assert_eq!(acc.sum, 1.0);
    }

    #[test]
    fn windows_follow_the_legacy_chunks_rule() {
        assert_eq!(plan_windows(10, 4, 8), vec![(0, 4), (4, 4), (8, 2)]);   // short tail >= 2 scored
        assert_eq!(plan_windows(9, 4, 8), vec![(0, 4), (4, 4)]);           // 1-token tail dropped
        assert_eq!(plan_windows(100, 4, 2), vec![(0, 4), (4, 4)]);         // max_windows
        assert!(plan_windows(1, 4, 8).is_empty());
        assert!(plan_windows(0, 4, 8).is_empty());
    }

    #[test]
    fn row_targets_shift_by_one_and_end_unscored() {
        let win = [10u32, 11, 12, 13];
        assert_eq!(row_targets(&win, 0, 4), vec![11, 12, 13, NO_TARGET]);
        // a chunk boundary inside the window: the last row of chunk 0 still has its target
        assert_eq!(row_targets(&win, 0, 2), vec![11, 12]);
        assert_eq!(row_targets(&win, 2, 4), vec![13, NO_TARGET]);
        // n-1 scored positions over any chunking
        let scored: usize = [(0usize, 3usize), (3, 4)].iter()
            .map(|&(a, b)| row_targets(&win, a, b).iter().filter(|&&t| t != NO_TARGET).count()).sum();
        assert_eq!(scored, win.len() - 1);
    }

    fn argv(s: &[&str]) -> Vec<String> { s.iter().map(|x| x.to_string()).collect() }

    #[test]
    fn flag_parsing_defaults_and_overrides() {
        let a = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t.txt"])).unwrap();
        assert_eq!((a.window, a.max_windows, a.max_pos), (1024, 8, 4096));
        assert_eq!(a.chunk, crate::exl3_serve::DEFAULT_PREFILL_CHUNK);
        assert_eq!(a.ref_windows, 0);
        let a = PplArgs::from_args(&argv(&["--perplexity", "--model-dir=/m", "--text", "t.txt", "--window", "2048",
            "--max-windows=5", "--max-seq-len", "4096", "--prefill-chunk", "512", "--ppl-ref-windows", "1",
            "--ppl-csv", "o.csv", "--ppl-dump-ids", "i.json", "--ppl-label", "x"])).unwrap();
        assert_eq!((a.window, a.max_windows, a.max_pos, a.chunk, a.ref_windows), (2048, 5, 4096, 512, 1));
        assert_eq!((a.csv.as_deref(), a.dump_ids.as_deref(), a.label.as_str()), (Some("o.csv"), Some("i.json"), "x"));
    }

    // ---- S-B9-REL-PPL2 W4: the registered options and what the tool refuses ----------------------

    const TOOL_FLAGS: [&str; 12] = ["perplexity", "text", "window", "max-windows", "ppl-ids", "ppl-csv",
        "ppl-dump-ids", "ppl-ref-windows", "ppl-label", "ppl-min-top1", "ppl-min-ppl", "ppl-max-ppl"];

    #[test]
    fn every_perplexity_option_is_registered_typed_and_listed_once() {
        let help = crate::opts::help_advanced();
        let diag = crate::opts::help_text(true);
        for f in TOOL_FLAGS {
            let id = crate::opts::lookup(f).unwrap_or_else(|| panic!("--{f} is not registered"));
            let o = id.def();
            assert_eq!(o.env, "", "--{f}: a new option has no environment variable");
            assert_eq!(o.cat, crate::opts::Cat::Ppl, "--{f}");
            assert_eq!(o.scope, crate::opts::Scope::Local, "--{f}: one process, never shipped to a node");
            assert!(!matches!(o.ty, crate::opts::Ty::Internal(_)), "--{f} must be a command-line option");
            // listed exactly once in --help, and not in --help-diag
            let lines = help.lines().filter(|l| l.trim_start().starts_with(&format!("--{f} ")) || l.trim() == format!("--{f}")).count();
            assert_eq!(lines, 1, "--{f} must be listed once in --help ({lines})");
            assert!(!diag.contains(&format!("    --{f} ")), "--{f} must not be a --help-diag option");
        }
        // typed with ranges: integers are ranged, paths and the label are text
        use crate::opts::Ty;
        assert_eq!(crate::opts::lookup("window").unwrap().def().ty, Ty::IntRange(1, i64::MAX));
        assert_eq!(crate::opts::lookup("max-windows").unwrap().def().ty, Ty::IntRange(1, i64::MAX));
        assert_eq!(crate::opts::lookup("ppl-ref-windows").unwrap().def().ty, Ty::IntRange(0, i64::MAX));
        assert_eq!(crate::opts::lookup("perplexity").unwrap().def().ty, Ty::Flag);
        for f in ["text", "ppl-ids", "ppl-csv", "ppl-dump-ids"] {
            assert_eq!(crate::opts::lookup(f).unwrap().def().ty, Ty::Path, "--{f}");
        }
        // every registered --ppl-* name is one of ours (the refusal's list is the registry's)
        let mut reg: Vec<&str> = ppl_flag_names();
        reg.sort();
        let mut want: Vec<&str> = TOOL_FLAGS.iter().copied().filter(|f| f.starts_with("ppl-")).collect();
        want.sort();
        assert_eq!(reg, want);
    }

    #[test]
    fn registry_ranges_refuse_through_parse_cli() {
        // The first bad option ends the parse before anything is written to the process-global store.
        let bad = |a: &[&str]| crate::opts::parse_cli(&argv(a));
        assert_eq!(bad(&["x", "--window", "0"]).unwrap_err(), "--window takes an integer >= 1 (got '0')");
        assert_eq!(bad(&["x", "--max-windows", "0"]).unwrap_err(), "--max-windows takes an integer >= 1 (got '0')");
        assert_eq!(bad(&["x", "--ppl-ref-windows", "-1"]).unwrap_err(), "--ppl-ref-windows takes an integer >= 0 (got '-1')");
        assert!(bad(&["x", "--window", "abc"]).is_err());
        assert!(bad(&["x", "--window=1.5"]).is_err());
        assert!(bad(&["x", "--ppl-csv"]).is_err(), "a path option needs a value");
        assert!(bad(&["x", "--ppl-min-top1", "high"]).is_err(), "a float option takes a number");
        // the pure parser agrees with parse_cli on every one of them (it is the same walk)
        assert_eq!(crate::opts::parse_args(&argv(&["x", "--window", "0"])).unwrap_err(),
                   "--window takes an integer >= 1 (got '0')");
        // and a good line is typed, canonical and last-wins
        let v = crate::opts::parse_args(&argv(&["x", "--window", "512", "--window=+1024", "--max-windows", "3"])).unwrap();
        let id = crate::opts::lookup("window").unwrap();
        assert_eq!(v.iter().rev().find(|(i, _)| *i == id).unwrap().1.as_deref(), Some("1024"));
    }

    #[test]
    fn an_unknown_ppl_name_is_refused_not_dropped() {
        // parse_cli alone skips a name it does not know: that is the defect (the reference was dropped)
        assert!(crate::opts::parse_cli(&argv(&["x", "--ppl-ref-windowz", "3"])).is_ok());
        let base = ["--perplexity", "--model-dir", "/m", "--text", "t"];
        let with = |extra: &[&str]| { let mut v: Vec<&str> = base.to_vec(); v.extend_from_slice(extra); PplArgs::from_args(&argv(&v)) };
        let e = format!("{:#}", with(&["--ppl-ref-windowz", "3"]).unwrap_err());
        assert!(e.contains("unknown option --ppl-ref-windowz") && e.contains("Did you mean --ppl-ref-windows?"), "{e}");
        let e = format!("{:#}", with(&["--ppl-bogus=1"]).unwrap_err());
        assert!(e.contains("unknown option --ppl-bogus") && e.contains("--ppl-csv"), "{e}");
        assert!(!e.contains("Did you mean"), "no suggestion for a name that is not close to any: {e}");
        assert!(with(&["--ppl"]).is_err(), "the bare namespace name is not an option either");
        // the registered names are all accepted
        assert!(with(&["--ppl-ref-windows", "1", "--ppl-csv", "o.csv", "--ppl-dump-ids", "i.json", "--ppl-label", "x",
                       "--ppl-min-top1", "0.3", "--ppl-min-ppl", "1.5", "--ppl-max-ppl", "50"]).is_ok());
        // a name that merely starts like another tool's flag is not in this namespace
        assert!(with(&["--prefill-chunk", "512"]).is_ok());
    }

    #[test]
    fn kv_cache_refusals() {
        let base = ["--perplexity", "--model-dir", "/m", "--text", "t"];
        let with = |kv: &str| { let mut v: Vec<&str> = base.to_vec(); v.extend_from_slice(&["--kv-cache", kv]); PplArgs::from_args(&argv(&v)) };
        for ok in ["f32", "f16", "fp8", "q8"] { assert!(with(ok).is_ok(), "{ok}"); }
        let e = format!("{:#}", with("q4x").unwrap_err());
        assert!(e.contains("q4x") && e.contains("deliberately broken"), "{e}");
        for nv in ["bf16", "q4", "tq", "tq3", "k8v4", "k8v8"] {
            let e = format!("{:#}", with(nv).unwrap_err());
            assert!(e.contains(&format!("--kv-cache {nv} is an NVFP4 KV format")), "{nv}: {e}");
        }
        assert!(with("int4").is_err());
        assert!(PplArgs::from_args(&argv(&base)).is_ok(), "no --kv-cache = the served default");
    }

    #[test]
    fn text_and_ppl_ids_are_alternatives() {
        let both = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t.txt", "--ppl-ids", "i.json"]));
        let e = format!("{:#}", both.unwrap_err());
        assert!(e.contains("--text and --ppl-ids are alternatives"), "{e}");
        let ids = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--ppl-ids", "i.json"])).unwrap();
        assert_eq!((ids.text.as_deref(), ids.ids.as_deref()), (None, Some("i.json")));
        assert!(PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m"])).is_err());
    }

    #[test]
    fn defaults_are_the_registry_rows() {
        let a = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t"])).unwrap();
        assert_eq!((a.window, a.max_windows, a.ref_windows, a.label.as_str()), (1024, 8, 0, "ppl"));
        assert_eq!(a.bounds, RunBounds::defaults());
        assert_eq!(a.bounds, RunBounds { min_ppl: 1.05, max_ppl: 1000.0, min_top1: 0.02 });
        for (flag, bad) in [("--ppl-min-top1", "2"), ("--ppl-min-top1", "-0.1"), ("--ppl-min-ppl", "0.5"),
                            ("--ppl-max-ppl", "1.0"), ("--ppl-min-ppl", "nan"), ("--ppl-max-ppl", "inf")] {
            let r = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t", flag, bad]));
            assert!(r.is_err(), "{flag} {bad}");
        }
    }

    // ---- S-B9-REL-PPL2 W5: each check red once on a deliberately broken input, then green -------

    const V: usize = 40; // a toy vocabulary row

    /// a distinct, zero-free toy row (f16 bit patterns 0x3c00.. upward)
    fn real(seed: u16) -> Vec<u16> { (0..V as u16).map(|i| 0x3c00 + seed * 64 + i).collect() }
    fn cat(rs: &[Vec<u16>]) -> Vec<u16> { rs.iter().flatten().copied().collect() }

    #[test]
    fn group_scan_red_on_unwritten_sparse_and_duplicated_rows_green_on_real_ones() {
        let (a, b) = (real(1), real(2));
        // GREEN: distinct rows with no zeros
        assert_eq!(group_defects(&cat(&[a.clone(), b.clone(), real(3)]), V, 3), GroupDefects::default());
        // GREEN: a legitimate row may carry a few zeros (padding columns): 2 of 40 is 5%, not "more than"
        let mut few = real(4);
        few[0] = 0; few[7] = 0;
        assert_eq!(group_defects(&cat(&[a.clone(), few]), V, 2), GroupDefects::default());
        // RED 1: a kernel that wrote nothing leaves the zero poison: an all-zero row
        let d = group_defects(&cat(&[a.clone(), vec![0; V], b.clone()]), V, 3);
        assert_eq!(d, GroupDefects { unwritten: 1, sparse: 0, duplicate: 0 });
        // RED 2: the whole group unwritten (a GEMM that no-ops at this width): every row, and each
        // zero row equals the one before it
        let d = group_defects(&cat(&[vec![0; V], vec![0; V], vec![0; V], vec![0; V]]), V, 4);
        assert_eq!(d, GroupDefects { unwritten: 4, sparse: 0, duplicate: 3 });
        // RED 3: part of the grid never ran: a long run of zeros inside a written row
        let mut half = real(5);
        for x in half.iter_mut().take(V / 2) { *x = 0; }
        assert_eq!(group_defects(&cat(&[a.clone(), half]), V, 2), GroupDefects { unwritten: 0, sparse: 1, duplicate: 0 });
        // RED 4: stale / broadcast output: two neighbours bit-identical
        let d = group_defects(&cat(&[a.clone(), a.clone(), b.clone()]), V, 3);
        assert_eq!(d, GroupDefects { unwritten: 0, sparse: 0, duplicate: 1 });
        // only the first `m` rows are looked at (the tail of the staging buffer is stale by design)
        assert_eq!(group_defects(&cat(&[a.clone(), b.clone(), vec![0; V]]), V, 2), GroupDefects::default());
        // a single row: nothing to compare with, but an unwritten row is still seen
        assert_eq!(group_defects(&vec![0; V], V, 1), GroupDefects { unwritten: 1, sparse: 0, duplicate: 0 });
    }

    #[test]
    fn run_bounds_red_on_the_v1_garbage_and_on_a_leak_green_on_the_real_cells() {
        let b = RunBounds::defaults();
        // RED: the first version's 4.05 cell (a 6-bit lm_head through a kernel that wrote nothing)
        let e = check_top1(51175, 0.0003, &b).unwrap_err();
        assert!(e.contains("top-1 accuracy 0.0003") && e.contains("NOT valid"), "{e}");
        let e = check_ppl(724390.9645, &b).unwrap_err();
        assert!(e.contains("outside 1.05 .. 1000"), "{e}");
        // RED: a perplexity near 1 = the model was told its answers (causal-mask / offset defect)
        assert!(check_ppl(1.0003, &b).is_err());
        // RED: not a number
        assert!(check_ppl(f64::NAN, &b).is_err() && check_ppl(f64::INFINITY, &b).is_err());
        assert!(check_top1(1000, f64::NAN, &b).is_err());
        // RED: the floor binds at its edge, not at zero
        assert!(check_top1(1000, 0.0199, &b).is_err());
        // GREEN: the 18 cells of the first session (PPL 2.31 .. 6.42, top-1 0.60 .. 0.77)
        for (ppl, t1) in [(3.5264, 0.7465), (6.3994, 0.6087), (2.3135, 0.7691), (3.5409, 0.7468), (6.4184, 0.6108),
                          (2.3179, 0.7698), (6.4239, 0.6104), (3.5355, 0.7466)] {
            check_ppl(ppl, &b).unwrap();
            check_top1(51175, t1, &b).unwrap();
        }
        // a handful of tokens says nothing about a model: the floor waits for TOP1_MIN_SAMPLE
        assert!(check_top1(TOP1_MIN_SAMPLE - 1, 0.0, &b).is_ok());
        assert!(check_top1(TOP1_MIN_SAMPLE, 0.0, &b).is_err());
        // tighter, class-specific bounds from a harness (code-like text)
        let code = RunBounds { min_ppl: 1.5, max_ppl: 12.0, min_top1: 0.5 };
        assert!(check_ppl(3.5264, &code).is_ok() && check_top1(51175, 0.7465, &code).is_ok());
        assert!(check_ppl(13.0, &code).is_err() && check_top1(51175, 0.45, &code).is_err());
        // --ppl-min-top1 0 switches the floor off, by the user's choice
        let off = RunBounds { min_top1: 0.0, ..b };
        assert!(check_top1(1000, 0.0, &off).is_ok());
    }

    // ---- S-B9-REL-PPL2 1b: a failing check ENDS A RUN (synthetic device / windows, no GPU) ------

    /// A synthetic lm_head: `buf` plays the device logits buffer. `stale_init` models uninitialised
    /// device memory (`alloc_zeros` does not zero): distinct non-zero rows.
    #[derive(Clone, Copy, PartialEq)]
    enum Fault { None, WritesNothing, WritesHalfTheColumns, BroadcastsRow0, WidthDiffers }

    struct FakeHead {
        v: usize,
        buf: Vec<u16>,
        fault: Fault,
        poison_calls: usize,
        groups: usize,
    }

    fn healthy_val(r: usize, j: usize) -> u16 { 0x3c00 + ((r * 13 + j * 5) % 400) as u16 }

    impl FakeHead {
        fn new(v: usize, r_max: usize, fault: Fault) -> FakeHead {
            let buf = (0..r_max * v).map(|i| 0x3d00 + (i % 200) as u16).collect(); // stale, non-zero
            FakeHead { v, buf, fault, poison_calls: 0, groups: 0 }
        }
    }

    impl HeadDevice for FakeHead {
        fn poison(&mut self, _m: usize) -> Result<()> {
            self.poison_calls += 1;
            self.buf.iter_mut().for_each(|x| *x = 0);
            Ok(())
        }
        fn run_group(&mut self, r0: usize, m: usize, host: &mut [u16]) -> Result<()> {
            self.groups += 1;
            let v = self.v;
            for r in 0..m {
                for j in 0..v {
                    let val = match self.fault {
                        Fault::WritesNothing => continue,
                        Fault::WritesHalfTheColumns if j >= v / 2 => continue,
                        Fault::BroadcastsRow0 => healthy_val(r0, j),
                        _ => healthy_val(r0 + r, j),
                    };
                    self.buf[r * v + j] = val;
                }
            }
            host[..self.buf.len()].copy_from_slice(&self.buf);
            Ok(())
        }
        fn run_single(&mut self, r0: usize, out: &mut [u16]) -> Result<()> {
            for j in 0..self.v {
                out[j] = healthy_val(r0, j) + (self.fault == Fault::WidthDiffers) as u16;
            }
            Ok(())
        }
        fn bits(&self) -> i32 { 6 }
    }

    fn head_state(v: usize, r_max: usize, targets: Vec<u32>) -> HeadState {
        let mut st = HeadState::new(r_max, 8, v);
        st.targets = targets;
        st
    }

    #[test]
    fn a_head_that_writes_nothing_ends_the_run_and_the_poison_is_what_shows_it() {
        let (v, r_max) = (64usize, 4usize);
        let targets: Vec<u32> = (0..10).map(|i| i as u32).collect();
        // GREEN: a healthy head scores every row, and the width check ran and agreed
        let mut dev = FakeHead::new(v, r_max, Fault::None);
        let mut st = head_state(v, r_max, targets.clone());
        head_chunk(&mut dev, &mut st, 10).unwrap();
        assert_eq!(st.acc.n, 10);
        assert_eq!(st.counters.groups, 3);                      // 4 + 4 + 2 rows
        assert_eq!(dev.poison_calls, 3, "every group is poisoned before its launch");
        assert_eq!(st.counters.width_check, Some((0, 4)));
        assert_eq!((st.counters.unwritten_rows, st.counters.sparse_rows, st.counters.duplicate_rows), (0, 0, 0));
        // RED: the launch writes nothing; the buffer held stale (distinct, non-zero) rows. Only the
        // poison turns that into the all-zero rows the scan refuses.
        let mut dev = FakeHead::new(v, r_max, Fault::WritesNothing);
        let mut st = head_state(v, r_max, targets.clone());
        let e = head_chunk(&mut dev, &mut st, 10).unwrap_err().to_string();
        assert!(e.contains("were never written") && e.contains("NOT valid") && e.contains("lm_head bits 6"), "{e}");
        assert_eq!(st.counters.groups, 1, "the run stops at the first group");
        assert_eq!(st.acc.n, 0, "nothing of the broken group was scored");
        // RED: only part of the grid ran
        let mut dev = FakeHead::new(v, r_max, Fault::WritesHalfTheColumns);
        let e = head_chunk(&mut dev, &mut head_state(v, r_max, targets.clone()), 10).unwrap_err().to_string();
        assert!(e.contains("mostly zero"), "{e}");
        // RED: row 0 broadcast into every row
        let mut dev = FakeHead::new(v, r_max, Fault::BroadcastsRow0);
        let e = head_chunk(&mut dev, &mut head_state(v, r_max, targets.clone()), 10).unwrap_err().to_string();
        assert!(e.contains("bit-identical"), "{e}");
        // RED: the m = 1 recomputation disagrees with the group
        let mut dev = FakeHead::new(v, r_max, Fault::WidthDiffers);
        let e = head_chunk(&mut dev, &mut head_state(v, r_max, targets), 10).unwrap_err().to_string();
        assert!(e.contains("width check FAILED"), "{e}");
        // a group with no scored target is not launched at all
        let mut dev = FakeHead::new(v, r_max, Fault::WritesNothing);
        let mut st = head_state(v, r_max, vec![NO_TARGET; 4]);
        head_chunk(&mut dev, &mut st, 4).unwrap();
        assert_eq!((dev.poison_calls, dev.groups, st.acc.n), (0, 0, 0));
    }

    #[test]
    fn the_poison_plan_covers_what_every_stage_of_every_head_class_writes() {
        // Every stage output of the chain must be zeroed before the launch, for the fused 3/4/5-bit
        // head, the unfused 6-bit head (exl3_hmma_gemm -> y_raw -> exl3_had_svh: the M1 hole of the
        // part-1 review, where only the final buffer was zeroed) and a split-K head.
        for (fused, ks) in [(true, 1u32), (false, 1), (true, 2), (false, 4)] {
            let plan = head_poison_plan(ks);
            for out in chain_stage_outputs(fused, ks) {
                assert!(plan.contains(&out), "fused={fused} ks={ks}: the chain writes {out:?} but the plan does not zero it");
            }
        }
        assert!(chain_stage_outputs(false, 1).contains(&HeadBuf::YRaw));
        assert!(!chain_stage_outputs(true, 1).contains(&HeadBuf::YRaw), "the fused path never touches y_raw");
        assert!(chain_stage_outputs(false, 2).contains(&HeadBuf::KsWs) && !head_poison_plan(1).contains(&HeadBuf::KsWs));
        assert!(head_poison_plan(1).contains(&HeadBuf::Logits));
    }

    fn acc_of(n: usize, top1: f64, mean_nll: f64) -> PplAcc {
        PplAcc { sum: mean_nll * n as f64, n, top1: (top1 * n as f64).round() as usize, ..Default::default() }
    }

    fn code_bounds() -> RunBounds { RunBounds { min_ppl: 1.5, max_ppl: 12.0, min_top1: 0.5 } }

    #[test]
    fn one_garbage_window_among_good_ones_fails_the_run() {
        let mut j = RunJudge::new(code_bounds());
        for wi in 1..=24 {
            j.after_window(wi, 2047, &acc_of(2047, 0.75, 1.26)).unwrap();      // code-like: PPL 3.5, top-1 0.75
        }
        // arithmetic of the review: the garbage window barely moves the running totals ...
        let mut cumulative = j.total.clone();
        cumulative.merge(&acc_of(2047, 0.001, 13.4));
        assert!(cumulative.top1_frac() > 0.7 && cumulative.ppl() < 6.0, "{} {}", cumulative.top1_frac(), cumulative.ppl());
        assert!(check_top1(cumulative.n, cumulative.top1_frac(), &code_bounds()).is_ok());
        assert!(check_ppl(cumulative.ppl(), &code_bounds()).is_ok(), "the mean alone passes both bounds");
        // ... so the window is judged on its own
        let e = j.after_window(25, 2047, &acc_of(2047, 0.001, 13.4)).unwrap_err();
        assert!(e.contains("window 25 (alone)") && e.contains("top-1 accuracy 0.0010"), "{e}");
        // and so is a window that is only too perplexing (top-1 fine)
        let mut j = RunJudge::new(code_bounds());
        let e = j.after_window(1, 2047, &acc_of(2047, 0.9, 7.0)).unwrap_err();     // PPL 1097 > 12
        assert!(e.contains("window 1 (alone)") && e.contains("above --ppl-max-ppl 12"), "{e}");
    }

    #[test]
    fn a_broken_head_is_known_after_its_first_window() {
        let mut j = RunJudge::new(RunBounds::defaults());
        let e = j.after_window(1, 2047, &acc_of(2047, 0.0003, 13.49)).unwrap_err();
        assert!(e.contains("window 1"), "{e}");
        // windows too short for the per-window reading: the running total takes over
        let mut j = RunJudge::new(RunBounds::defaults());
        assert!(j.after_window(1, 31, &acc_of(31, 0.0, 12.0)).is_ok(), "31 tokens: no reading yet");
        assert!(j.after_window(2, 31, &acc_of(31, 0.0, 12.0)).is_ok(), "62 tokens: no reading yet");
        let e = j.after_window(3, 31, &acc_of(31, 0.0, 12.0)).unwrap_err();
        assert!(e.contains("after window 3"), "{e}");
        // a window that scored the wrong number of positions, or non-finite terms, fails by itself
        let mut j = RunJudge::new(RunBounds::defaults());
        assert!(j.after_window(1, 2047, &acc_of(2000, 0.7, 1.0)).unwrap_err().contains("2000 positions scored"));
        let mut bad = acc_of(2047, 0.7, 1.0);
        bad.nonfinite_nll = 1;
        assert!(j.after_window(1, 2047, &bad).unwrap_err().contains("non-finite"));
    }

    #[test]
    fn the_final_bounds_end_a_run_every_window_of_which_passed() {
        // PPL ~ 1.0003 in every window (the model was told its answers), top-1 0.99: no window is
        // above the ceiling and none below the top-1 floor; only the conclusion can say no.
        let mut j = RunJudge::new(RunBounds::defaults());
        for wi in 1..=3 { j.after_window(wi, 2047, &acc_of(2047, 0.99, 0.0003)).unwrap(); }
        let head = HeadCounters { groups: 384, ..Default::default() };
        let e = j.conclude(&head).unwrap_err();
        assert!(e.contains("perplexity 1.0003") && e.contains("outside 1.05 .. 1000"), "{e}");
        // the path is cut: a run that never launched the head cannot conclude
        let mut j = RunJudge::new(RunBounds::defaults());
        j.after_window(1, 2047, &acc_of(2047, 0.75, 1.26)).unwrap();
        assert!(j.conclude(&HeadCounters::default()).unwrap_err().contains("never ran"));
        // GREEN: the first session's real cells conclude, and say whether the floor applied
        for (ppl, t1) in [(3.5264f64, 0.7465), (6.3994, 0.6087), (2.3135, 0.7691)] {
            let mut j = RunJudge::new(RunBounds::defaults());
            j.after_window(1, 2047, &acc_of(2047, t1, ppl.ln())).unwrap();
            let s = j.conclude(&head).unwrap();
            assert!((s.ppl - ppl).abs() < 1e-9 && s.top1_floor_applied, "{s:?}");
        }
        // a run under 64 tokens concludes without a top-1 floor and SAYS so
        let mut j = RunJudge::new(RunBounds::defaults());
        j.after_window(1, 9, &acc_of(9, 0.0, 3.0)).unwrap();
        let s = j.conclude(&head).unwrap();
        assert!(!s.top1_floor_applied);
    }

    struct FakeWindows {
        script: Vec<PplAcc>,
        scored: usize,
    }

    impl WindowSource for FakeWindows {
        fn score_window(&mut self, wi: usize, win: &[u32]) -> Result<WindowScore> {
            self.scored += 1;
            Ok(WindowScore { acc: self.script[wi].clone(), nll: vec![1.0; win.len()], secs: 0.0 })
        }
        fn counters(&self) -> HeadCounters { HeadCounters { groups: 10, rows_checked: 10, ..Default::default() } }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ppl2_run_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Drive the whole post-device path (score_all -> finish_run) over `script`, with outputs in a
    /// fresh temp dir. Returns (result, stdout text, dir).
    fn drive(tag: &str, script: Vec<PplAcc>, bounds: RunBounds) -> (Result<()>, String, std::path::PathBuf, usize) {
        let dir = temp_dir(tag);
        let n_win = script.len();
        let n = 5usize; // tokens per window; acc.n is whatever the script says
        let ids: Vec<u32> = (0..(n * n_win) as u32).collect();
        let wins: Vec<(usize, usize)> = (0..n_win).map(|i| (i * n, n)).collect();
        let a = PplArgs {
            dir: "/m".into(), text: Some("t".into()), ids: None, window: n, max_windows: n_win, max_pos: 4096,
            chunk: 4095, ref_windows: 0,
            csv: Some(dir.join("o.csv").to_string_lossy().into_owned()),
            dump_ids: Some(dir.join("ids.json").to_string_lossy().into_owned()),
            label: "t".into(), bounds,
        };
        let mut src = FakeWindows { script: script.into_iter().map(|mut a| { a.n = a.n; a }).collect(), scored: 0 };
        let mut judge = RunJudge::new(bounds);
        let mut out: Vec<u8> = Vec::new();
        // the fake windows report acc.n scored positions; the judge expects n - 1 = 4 per window
        let res = (|| {
            let (nll, failure) = score_all(&mut src, &wins, &ids, &mut judge, &mut out)?;
            let fi = FinishInputs { a: &a, kv_name: "q8", wins: &wins, ids: &ids, vocab: 1000, src: "t",
                                    per_window_nll: &nll, dec_nll: &[], head: src.counters(), elapsed_s: 0.0 };
            finish_run(&fi, &judge, failure, &mut out)
        })();
        (res, String::from_utf8(out).unwrap(), dir, src.scored)
    }

    fn w4(top1_hits: usize, nll: f64) -> PplAcc {
        // 4 scored positions per window (n = 5 tokens), floor-exempt (< 64): only the conclusion judges
        PplAcc { sum: nll * 4.0, n: 4, top1: top1_hits, ..Default::default() }
    }

    #[test]
    fn a_failed_run_prints_no_result_line_and_owns_no_output_name() {
        let b = RunBounds::defaults();
        // GREEN: three plausible windows -> RESULT once, csv and ids under their real names
        let (r, out, dir, scored) = drive("ok", vec![w4(3, 1.2), w4(3, 1.3), w4(2, 1.1)], b);
        r.unwrap();
        assert_eq!(scored, 3);
        assert_eq!(out.matches("RESULT: tokens=12").count(), 1, "{out}");
        assert!(out.contains("PPL_HEAD_CHECK groups=10") && out.contains("PPL_BOUNDS"), "{out}");
        assert!(dir.join("o.csv").exists() && dir.join("ids.json").exists());
        assert!(!dir.join("o.csv.INVALID").exists());
        let csv = std::fs::read_to_string(dir.join("o.csv")).unwrap();
        assert!(csv.starts_with("label,window,pos,target,nll,nll_decode"), "{csv}");

        // RED (final bounds): every window has PPL ~ 1.0 -> no window is out of range alone, the run is
        let (r, out, dir, scored) = drive("leak", vec![w4(4, 0.0003), w4(4, 0.0003)], b);
        let e = r.unwrap_err().to_string();
        assert!(e.contains("perplexity 1.0003") && scored == 2, "{e}");
        assert!(!out.contains("RESULT:"), "a failed run prints no RESULT line:\n{out}");
        assert!(!dir.join("o.csv").exists() && !dir.join("ids.json").exists(), "no output under the real names");
        let inv = std::fs::read_to_string(dir.join("o.csv.INVALID")).unwrap();
        assert!(inv.starts_with("# INVALID RUN - perplexity 1.0003"), "{inv}");
        assert!(out.contains("PPL_HEAD_CHECK") && out.contains("PPL_STATS"), "the diagnostics are still printed");

        // RED (a window): the second window is the model's output of an lm_head that is not this
        // model's; the run stops there (the third window is never scored)
        let bad = PplAcc { sum: 13.0 * 80.0, n: 80, top1: 0, ..Default::default() };
        let mut script = vec![acc_of(80, 0.75, 1.2), bad, acc_of(80, 0.75, 1.2)];
        // windows are judged against n - 1 = 4 positions: give the judge's expectation what the script says
        for a in script.iter_mut() { a.n = 4; }
        script[1] = PplAcc { sum: 13.0 * 4.0, n: 4, top1: 0, ..Default::default() };
        let (r, out, dir, scored) = drive("bad_window", script, b);
        // 4 tokens are below the per-window gate, so the conclusion decides: PPL e^(mean of 1.2, 13, 1.2)
        let e = r.unwrap_err().to_string();
        assert!(!out.contains("RESULT:") && scored == 3, "{e}\n{out}");
        assert!(!dir.join("o.csv").exists() && dir.join("o.csv.INVALID").exists());
    }

    #[test]
    fn a_window_judged_alone_stops_the_run_where_it_lands() {
        let b = RunBounds::defaults();
        // 65 scored positions per window so the per-window gate applies
        let big = |hits: usize, nll: f64| PplAcc { sum: nll * 65.0, n: 65, top1: hits, ..Default::default() };
        let dir = temp_dir("stop");
        let n = 66usize;
        let ids: Vec<u32> = (0..(n * 3) as u32).collect();
        let wins: Vec<(usize, usize)> = (0..3).map(|i| (i * n, n)).collect();
        let mut src = FakeWindows { script: vec![big(48, 1.2), big(0, 13.0), big(48, 1.2)], scored: 0 };
        let mut judge = RunJudge::new(b);
        let mut out: Vec<u8> = Vec::new();
        let (nll, failure) = score_all(&mut src, &wins, &ids, &mut judge, &mut out).unwrap();
        assert_eq!(src.scored, 2, "the third window is never scored");
        assert_eq!(nll.len(), 2);
        assert!(failure.as_deref().unwrap().contains("window 2 (alone)"), "{failure:?}");
        let _ = dir;
    }

    // ---- 1b item 8 / 6: the refusals added to the tool ---------------------------------------

    #[test]
    fn a_value_less_path_option_that_would_swallow_the_next_flag_is_refused() {
        let tool = |extra: &[&str]| {
            let mut v = vec!["--perplexity", "--model-dir", "/m"];
            v.extend_from_slice(extra);
            PplArgs::from_args(&argv(&v))
        };
        let e = format!("{:#}", tool(&["--text", "--mtp", "off"]).unwrap_err());
        assert!(e.contains("--text needs a value") && e.contains("--mtp") && e.contains("swallow"), "{e}");
        let e = format!("{:#}", tool(&["--text", "t", "--ppl-csv", "--window", "2048"]).unwrap_err());
        assert!(e.contains("--ppl-csv needs a value") && e.contains("--window"), "{e}");
        assert!(tool(&["--ppl-label", "--ppl-min-top1", "0.1", "--text", "t"]).is_err());
        let e = format!("{:#}", PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "--kv-cache", "q8", "--text", "t"])).unwrap_err());
        assert!(e.contains("--model-dir needs a value"), "{e}");
        // an explicit `=` cannot swallow anything; a normal line passes
        assert!(tool(&["--text=--odd-name.txt"]).is_ok());
        assert!(tool(&["--text", "t.txt", "--ppl-csv", "o.csv", "--window", "512"]).is_ok());
    }

    #[test]
    fn a_repeated_option_with_different_values_is_refused() {
        let tool = |extra: &[&str]| {
            let mut v = vec!["--perplexity", "--model-dir", "/m", "--text", "t"];
            v.extend_from_slice(extra);
            PplArgs::from_args(&argv(&v))
        };
        // validated as q8, measured as q4x (the bridge reads the first): refused instead
        let e = format!("{:#}", tool(&["--kv-cache", "q4x", "--kv-cache", "q8"]).unwrap_err());
        assert!(e.contains("--kv-cache is given twice with different values ('q4x' and 'q8')"), "{e}");
        assert!(tool(&["--window", "1024", "--window=2048"]).is_err());
        assert!(tool(&["--ppl-label", "a", "--ppl-label", "b"]).is_err());
        // the same value twice is not a conflict
        assert!(tool(&["--kv-cache", "q8", "--kv-cache=q8", "--window", "512", "--window", "512"]).is_ok());
        // the mode flag itself may repeat
        assert!(tool(&["--perplexity"]).is_ok());
    }

    #[test]
    fn the_unknown_name_hint_lists_every_option_of_the_tool() {
        let e = format!("{:#}", PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t", "--ppl-zzzz-qqqq", "1"])).unwrap_err());
        for f in ["--perplexity", "--text", "--window", "--max-windows", "--ppl-ids", "--ppl-max-ppl", "--model-dir"] {
            assert!(e.contains(f), "{f} missing from: {e}");
        }
    }

    #[test]
    fn flag_parsing_rejects_nonsense() {
        let base = ["--perplexity", "--model-dir", "/m", "--text", "t"];
        let with = |extra: &[&str]| { let mut v: Vec<&str> = base.to_vec(); v.extend_from_slice(extra); PplArgs::from_args(&argv(&v)) };
        assert!(with(&["--window", "abc"]).is_err());
        assert!(with(&["--window", "1"]).is_err());
        assert!(with(&["--window", "8192"]).is_err(), "window > max-seq-len must be refused");
        assert!(with(&["--max-windows", "0"]).is_err());
        assert!(PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m"])).is_err(), "no text");
        assert!(PplArgs::from_args(&argv(&["--perplexity", "--text", "t"])).is_err(), "no model dir");
    }
}
