//! `--draft-vocab` (FR-Spec): the row subset the MTP drafter may propose from.
//! Siblings: `mod.rs` holds the directory map.

use super::*;

// ================================ DRAFT-VOCAB row subset ================================
//
// Picking a draft token needs an argmax over the vocabulary, i.e. a second full read of the LM head.
// On 9B that is 572 MB -- **2.75 ms, 11% of a decode step -- and it is paid (depth-1) times per
// speculative step**, which makes it essentially the entire slope of r(d). It is the single biggest
// cost left in speculation.
//
// The fix (FR-Spec): give the DRAFTER a smaller LM head. Rank the vocabulary by corpus frequency,
// keep the top slice, and let the drafter propose only from that. The VERIFY keeps the full head, so:
//
//   * **Greedy stays exactly lossless.** The proposal mechanism only affects how OFTEN a draft
//     matches; every emitted token is still the full model's argmax. A token outside the subset simply
//     never gets proposed, so that position is always rejected -- costing acceptance, never correctness.
//   * **Stochastic stays distribution-exact.** If the drafter samples from the RENORMALIZED restricted
//     softmax, then that restricted distribution *is* `q`, and `min(1, p/q)` is the standard
//     Leviathan/Chen scheme with a perfectly valid proposal. Tokens outside the subset have q=0 and can
//     only enter through the residual `(p-q)+` resample -- which is over the FULL vocab on the verify
//     side, which we already pay for. No approximation anywhere.
//
// Rows are independent in both codecs (NVFP4 has per-row block scales + one tensor scale; FP8 has one
// scale per row), so a row subset is EXACT -- no requantization, no loss.

/// Take a subset of rows from a row-major NVFP4 tensor. Exact: rows do not interact.
pub fn subset_rows_nvfp4(qw: &[u8], sc: &[u8], k: usize, rows: &[u32]) -> (Vec<u8>, Vec<u8>) {
    let nblk = k / BLOCK;
    let (rb, sb) = (k / 2, nblk);
    let mut oq = Vec::with_capacity(rows.len() * rb);
    let mut os = Vec::with_capacity(rows.len() * sb);
    for &r in rows {
        let r = r as usize;
        oq.extend_from_slice(&qw[r * rb..(r + 1) * rb]);
        os.extend_from_slice(&sc[r * sb..(r + 1) * sb]);
    }
    (oq, os)
}

/// Take a subset of rows from a row-major FP8 tensor (row scales come along).
pub fn subset_rows_fp8(qw: &[u8], rs: &[f32], k: usize, rows: &[u32]) -> (Vec<u8>, Vec<f32>) {
    let mut oq = Vec::with_capacity(rows.len() * k);
    let mut os = Vec::with_capacity(rows.len());
    for &r in rows {
        let r = r as usize;
        oq.extend_from_slice(&qw[r * k..(r + 1) * k]);
        os.push(rs[r]);
    }
    (oq, os)
}

/// `--draft-vocab` (FR-Spec): how much of the vocabulary the MTP drafter may PROPOSE from. ON by
/// default for the Qwen3.5/3.6/3.8 family (`default_for`). The verify always scores the full
/// vocabulary, so the subset only trades acceptance (a token outside it is never drafted) against the
/// draft chain's LM-head bytes; greedy output is unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DraftVocab {
    /// the full vocabulary: no reduced draft head is built (`--draft-vocab off`; the default outside
    /// the Qwen3.5/3.6/3.8 family)
    Off,
    /// the `n` most frequent token ids (BPE id order), plus the special-token tail
    Top(usize),
    /// a fraction in (0, 1] of the vocabulary, same id order
    Frac(f64),
}

/// Rows `--draft-vocab on` keeps. Measured on prose+code, the top 65536 ids of the Qwen BPE vocabulary
/// (26%) cover 97.5% of emitted tokens.
pub const DRAFT_VOCAB_ON_ROWS: usize = 65536;

impl DraftVocab {
    /// The command-line spellings: `off` (also 0/full/false/no), `on` (= DRAFT_VOCAB_ON_ROWS rows; also
    /// true/yes), a row count `N`, a percentage `P%`, or a fraction `0.F`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "off" | "0" | "full" | "false" | "no" => return Ok(Self::Off),
            "on" | "true" | "yes" => return Ok(Self::Top(DRAFT_VOCAB_ON_ROWS)),
            _ => {}
        }
        let bad = || format!("--draft-vocab takes off|on|<rows>|<percent>%|<fraction> (got '{s}')");
        if let Some(p) = t.strip_suffix('%') {
            let p: f64 = p.trim().parse().map_err(|_| bad())?;
            return if p > 0.0 && p <= 100.0 { Ok(Self::Frac(p / 100.0)) } else { Err(bad()) };
        }
        if t.contains('.') {
            let f: f64 = t.parse().map_err(|_| bad())?;
            return if f > 0.0 && f <= 1.0 { Ok(Self::Frac(f)) } else { Err(bad()) };
        }
        match t.parse::<u64>() {
            Ok(n) if n > 0 => Ok(Self::Top(n as usize)),
            _ => Err(bad()),
        }
    }

    /// The registry's stored spelling (validated at boot; the head ships it to TP nodes verbatim).
    pub fn canonical_text(s: &str) -> Result<String, String> {
        let t = s.trim().to_ascii_lowercase();
        Ok(match Self::parse(&t)? {
            Self::Off => "off".to_string(),
            _ if matches!(t.as_str(), "on" | "true" | "yes") => "on".to_string(),
            _ => t,
        })
    }

    /// The setting when `--draft-vocab` is not given (owner 2026-10-27): ON for the Qwen3.5/3.6/3.8
    /// family, where it was measured (+7..+22% sampled decode, 0.8B..27B, TP=1/TP=2); OFF elsewhere
    /// (hy_v3 unverified; qwen4_exp never builds one). Non-Latin-script text is mostly outside the
    /// id-order subset (Chinese ~15% of tokens proposable): `--draft-vocab off` for those workloads.
    pub fn default_for(family: crate::qwen::Family) -> Self {
        match family {
            crate::qwen::Family::Qwen35 => Self::Top(DRAFT_VOCAB_ON_ROWS),
            _ => Self::Off,
        }
    }

    /// The most-frequent ids to keep for a vocabulary of `vocab` rows (0 = off).
    pub fn top(self, vocab: usize) -> usize {
        match self {
            Self::Off => 0,
            Self::Top(n) => n,
            Self::Frac(f) => ((f * vocab as f64).round() as usize).max(1),
        }
    }
}

/// The rows a draft head should keep: the `top` most-frequent tokens, PLUS the tail of the vocabulary.
///
/// The tail matters more than it looks. Special/added tokens (`<|im_end|>`, `<|endoftext|>`, the tool
/// and think markers) live at the TOP of the id range by convention, and the model must be able to
/// emit them -- `<|im_end|>` is how a chat turn STOPS. A drafter that can never propose it would fail
/// to draft the single most predictable token in the whole conversation.
///
/// Qwen's BPE ids are ordered by merge rank, which tracks training-corpus frequency (`Ġwould` is id
/// 1000; Thai and Arabic start around 150k). Measured on prose+code, the top 65536 ids (26% of the
/// vocabulary) cover **97.5%** of emitted tokens.
///
/// Result length is padded up to a multiple of 16 (the MMA tile height) with real rows, never dummies:
/// a zero row would have logit 0 and could WIN an argmax against all-negative logits.
pub fn draft_vocab_rows(top: usize, vocab: usize) -> Vec<u32> {
    const TAIL: usize = 512;                       // covers every special/added token, with margin
    // hy_v3's specials start at 120000 (`<｜hy_eos…｜>`, think/tool markers) — the generic
    // 512-row tail (120320+) would miss them all, and a drafter that cannot propose hy_eos
    // can never draft the end of a chat turn. Cover the whole special block for that vocab.
    let tail_start = if vocab == 120832 { 120000 } else { vocab.saturating_sub(TAIL) };
    let mut top = top.min(tail_start);
    // grow `top` (with real, more-frequent tokens) until the total is a multiple of 16
    while (top + (vocab - tail_start)) % MMA_M != 0 { top += 1; }
    let mut rows: Vec<u32> = (0..top as u32).collect();
    rows.extend(tail_start as u32..vocab as u32);
    rows
}

/// FR-Spec subset from an explicit id file (--draft-vocab-file) — the corpus-ranked
/// variant of `draft_vocab_rows` (the offline artifact step produces this from real frequencies).
/// Format: whitespace/newline-separated decimal token ids. The special-token tail is appended
/// (same rule as `draft_vocab_rows`, so a corpus list that omits specials stays chat-safe), then
/// id-order tokens fill the last partial 16-row tile. Rows are independent in every codec, so ANY
/// subset is exact — the file only chooses WHICH tokens are proposable.
pub fn draft_vocab_rows_file(path: &str, vocab: usize) -> std::io::Result<Vec<u32>> {
    let txt = std::fs::read_to_string(path)?;
    let mut rows: Vec<u32> = txt.split_whitespace()
        .filter_map(|t| t.parse::<u32>().ok())
        .filter(|&i| (i as usize) < vocab)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    const TAIL: usize = 512;
    let tail_start = if vocab == 120832 { 120000 } else { vocab.saturating_sub(TAIL) };
    let mut next: std::collections::HashSet<u32> = rows.iter().copied().collect();
    for i in tail_start as u32..vocab as u32 {
        if next.insert(i) { rows.push(i); }
    }
    let mut fill = 0u32;
    while rows.len() % MMA_M != 0 {
        while !next.insert(fill) { fill += 1; }
        rows.push(fill);
    }
    Ok(rows)
}
