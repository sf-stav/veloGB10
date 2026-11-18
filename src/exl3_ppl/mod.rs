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
//!
//! File map (this was the single `exl3_ppl.rs`; every item keeps its path): `mod.rs` = the perplexity driver on an EXL3 pack (chunked prefill with an all-rows lm_head, windows, scoring, report); `tests.rs` = the inline unit tests (module `tests`, same name).

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
mod tests;
