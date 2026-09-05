//! S5F3 — the draft-parity step dump (dump-only, default OFF; `--df2-step-dump <dir>` /
//! `GB10_DF2_STEP_DUMP=<dir>`).
//!
//! During a `--bench-df2-matrix --parity` run (or any `BatchScheduler` speculation run), one
//! JSONL record per verify step:
//!
//! * the 7 draft tokens + the selector candidate table / unary / chain scores (the path q is
//!   computable offline — softmax over the 16 scores at T=1.0, the SGLang `sample_path` q);
//! * the target's top-k-20-renorm p for the drafts (`p_of_draft`) **and** the full top-20
//!   (token, renorm-p) table per verify column (the `df2_topk20_dump_b` kernel — lets the
//!   analysis score ANY token, e.g. the oracle-replay drafts, and cross-checks the kernel's
//!   p_of_draft — the S3 p-fidelity check);
//! * accept/reject + the sampled alternative (resid) + the bonus token + nacc/emitted;
//! * the step's absolute position (+ the run's token stream, so the analysis can compute the
//!   inside/outside-`<think>` flag exactly as the harness's parity cuts do);
//! * per-tap-layer × per-column checksums (FNV-1a-64 over the raw bf16 bits) of the verify's
//!   fed-span taps, and the raw tap vectors accumulated per job into `taps.bin`
//!   (`[T, 25600]` f32 — the full committed history, so the S2F oracle replay can re-run ANY
//!   step's complete prefix: `tap_hiddens = taps[0..main_pos)`, `anchor = committed_tok`);
//! * for the first `raw_steps` steps, the round's `h_final` (`[8, 5120]` f32) in `hfinal.bin`
//!   (the S1 layer-bisect surface: oracle-vs-engine block hiddens on the same taps).
//!
//! The MTP lane writes the same verify-side fields (no taps/selector) — its healthy acceptance
//! is the in-engine p-computation control.
//!
//! Everything here is write-only: no engine behavior depends on the dump's presence.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::dflash2::capture::Df2PrimeSink;
use crate::dflash2::{BLOCK, HIDDEN, TAP_CONCAT_DIM, TAP_LAYERS};

/// The number of steps whose RAW h_final is retained (the oracle-replay window).
pub const RAW_STEPS: usize = 64;
/// The number of steps whose per-layer RING rows (ctx rows near C + the injected span) are
/// retained (the S1 ring-bisect window).
pub const RING_STEPS: usize = 8;
/// The top-k the dump kernel retains per verify column (the protocol's k20).
pub const DUMP_TOPK: usize = 20;

/// The per-step DFlash2 record (serialized to one JSONL line).
#[derive(Serialize)]
pub struct Df2StepRec {
    pub step: u64,
    /// The lane's absolute position at step entry (= the committed count, = the round's nprev).
    pub pos: usize,
    /// The anchor token (the round's block row 0 — the last committed token, at position pos-1).
    pub committed: u32,
    pub greedy: bool,
    pub realq: bool,
    /// The 7 draft tokens, block rows 1..8 at positions pos+1..pos+7.
    pub drafts: Vec<u32>,
    /// The target's top-k-20-renorm p of each draft (verify column j = position pos+1+j).
    pub p_draft: Vec<f32>,
    /// The rejection alternative per draft (a sample from p \ {draft}); empty on the greedy lane.
    pub resid: Vec<u32>,
    /// The all-accepted bonus token.
    pub bonus: u32,
    pub nacc: usize,
    pub emitted: usize,
    /// The selector's drawn-path q per position ([7]); 1.0 on the greedy lane (q computable
    /// offline from `scores`), the drawn multinomial weights on the real-q lane.
    pub q_rows: Vec<f32>,
    /// The top-16 candidate token ids per position, row-major [7][16].
    pub candidates: Vec<u32>,
    /// The selector's candidate q weights (softmax over the 16 candidates at the walk
    /// temperature, already normalized to sum to 1 per position), row-major [7][16]. D0's
    /// acceptance-ceiling instrument: A = Σ_k min(p(cand_k), q(cand_k)) per position.
    pub cand_q: Vec<f32>,
    /// The candidate unary logits, row-major [7][16].
    pub unary: Vec<f32>,
    /// The final chain scores, row-major [7][16] (the greedy path only).
    pub scores: Vec<f32>,
    /// The full top-20 (token, renorm-p) table per verify column j=0..7, packed u64
    /// `(p_bits << 32) | tok` — column j = the distribution at position pos+1+j.
    pub top20: Vec<u64>,
    /// Per (tap layer, column) checksums of the verify's fed-span taps, row-major [5][8].
    pub tap_ck: Vec<u64>,
    /// True when this step's raw h_final was written to hfinal.bin (steps < RAW_STEPS).
    pub hfinal_written: bool,

    // ---- PLAN/25 Phase 0 (coverage curves) — dump-only additions, all empty when the
    // ---- coverage capture is off.
    /// Greedy lane: per verify column j the TARGET's top-4 token ids + raw logits
    /// (row-major [8][4]), the full-vocab softmax p of the argmax, and the top1−top2
    /// logit gap (the near-tie gauge). Empty on the sample lanes (their distribution-side
    /// table is `top20`).
    pub tgt_ids: Vec<u32>,
    pub tgt_logit: Vec<f32>,
    pub tgt_p1: Vec<f32>,
    pub tgt_margin: Vec<f32>,
    /// The LOGGING-ONLY MTP chain run on the same step grid (the union instrument): per
    /// draft position j (1..7) the MTP head's top-8 candidate ids + raw logits, row-major
    /// [7][8], conditioned on the SAME committed prefix the DF2 round drafted from. The
    /// chain writes nothing but this record — the DF2 selection path is untouched.
    pub mtp_ids: Vec<u32>,
    pub mtp_logit: Vec<f32>,
}

/// The per-step MTP record (the p-computation control).
#[derive(Serialize)]
pub struct MtpStepRec {
    pub step: u64,
    pub pos: usize,
    pub committed: u32,
    pub depth: usize,
    pub drafts: Vec<u32>,
    pub p_draft: Vec<f32>,
    pub resid: Vec<u32>,
    pub bonus: u32,
    pub nacc: usize,
    pub emitted: usize,

    // ---- PLAN/25 Phase 0 (coverage curves) — dump-only additions, all empty when the
    // ---- coverage capture is off (the S5F3 parity runs leave them at their defaults).
    /// Per verify column j (position pos+1+j): the TARGET's top-4 token ids (row-major
    /// [n][4]) with raw logits, the full-vocab softmax p of the argmax, and the
    /// top1−top2 LOGIT gap (the near-tie gauge). Greedy lane only (the sample lane's
    /// distribution-side table is `tgt_top20`).
    pub tgt_ids: Vec<u32>,
    pub tgt_logit: Vec<f32>,
    pub tgt_p1: Vec<f32>,
    pub tgt_margin: Vec<f32>,
    /// Per draft position j: the DRAFTER head's top-8 candidate ids + raw logits
    /// (row-major [depth-1][8]). top-1 == the draft (argmax path unchanged — the head is
    /// re-run read-only for this readback, never used to draft).
    pub draft_topk_ids: Vec<u32>,
    pub draft_topk_logit: Vec<f32>,
    /// Sample lane: the verify's packed (p<<32|tok) top-20 per column (verify_forward_sample's
    /// `t20_out`), empty on the greedy lane.
    pub tgt_top20: Vec<u64>,
}

/// FNV-1a 64 over raw u16s (bf16 bit patterns).
fn fnv64(words: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in words {
        h ^= w as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// PLAN/25 Phase 0 — the target-side coverage readback (host, dump-only): top-`k` token ids
/// + raw logits from one verify column, the full-vocab softmax p of the argmax, and the
/// top1−top2 logit gap (the near-tie gauge). Same arithmetic as `bench_accept`'s inline
/// per-column scan (the Phase-0 methodology's reference). Overflow-safe on ±inf logits.
pub fn tgt_topk_host(col: &[f32], k: usize) -> (Vec<u32>, Vec<f32>, f32, f32) {
    let n = k.min(col.len()).max(1);
    let mut idx: Vec<usize> = (0..col.len()).collect();
    // Partial-select the top-n (O(len)), then order just those. The `.then(a.cmp(&b))`
    // tie-break keeps the LOWEST index among bit-equal logits — the same element the verify's
    // argmax kernel keeps — so ids[0] IS the device argmax, not a coin flip among equals.
    idx.select_nth_unstable_by(n - 1, |&a, &b| col[b].total_cmp(&col[a]));
    idx.truncate(n);
    idx.sort_by(|&a, &b| col[b].total_cmp(&col[a]).then(a.cmp(&b)));
    let ids: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
    let vals: Vec<f32> = idx.iter().map(|&i| col[i]).collect();
    let b1 = vals[0];
    let denom: f32 = col.iter().map(|&x| (x - b1).exp()).sum();
    let margin = vals[0] - vals.get(1).unwrap_or(&vals[0]);
    (ids, vals, 1.0 / denom, margin)
}

/// The dump writer. Owned by the `BatchScheduler` (one per process); jobs write sequentially.
pub struct StepDump {
    out_dir: PathBuf,
    jsonl: std::fs::File,
    /// Per-job accumulated raw taps, `[filled, TAP_CONCAT_DIM]` f32 row-major.
    taps: Vec<f32>,
    /// The first absolute position the current job's accumulation covers.
    taps_from: usize,
    /// Number of positions accumulated for the current job.
    taps_n: usize,
    /// The highest position the current job has reached (for the union-coverage bookkeeping).
    taps_max: usize,
    /// The current job's tag (prompt name + rep) — written at job_start.
    job_tag: String,
    job_plen: usize,
    hfinal: std::fs::File,
    ringrows: std::fs::File,
    n_steps: u64,
}

impl StepDump {
    pub fn new(dir: &str) -> std::io::Result<Self> {
        let out_dir = PathBuf::from(dir);
        std::fs::create_dir_all(&out_dir)?;
        let jsonl = std::fs::File::create(out_dir.join("steps.jsonl"))?;
        let hfinal = std::fs::File::create(out_dir.join("hfinal.bin"))?;
        let ringrows = std::fs::File::create(out_dir.join("ringrows.bin"))?;
        Ok(StepDump {
            out_dir,
            jsonl,
            taps: Vec::new(),
            taps_from: 0,
            taps_n: 0,
            taps_max: 0,
            job_tag: String::new(),
            job_plen: 0,
            hfinal,
            ringrows,
            n_steps: 0,
        })
    }

    pub fn out_dir(&self) -> &Path { &self.out_dir }

    /// Begin a job: read the prime sink's prompt taps (positions 0..plen) into the tap
    /// accumulation and write the job marker. The prime sink holds the LAST prefill window at
    /// columns [0, n) — the A6 prompts are single-window (plen <= 8192), so cols [0..plen)
    /// are the full prompt taps.
    pub fn job_start(&mut self, tag: &str, prompt: &[u32],
                     dev: &Arc<cudarc::driver::CudaDevice>) {
        let plen = prompt.len();
        self.job_tag = tag.to_string();
        self.job_plen = plen;
        self.taps.clear();
        self.taps_n = 0;
        self.taps_max = 0;
        self.taps_from = 0;
        let marker = format!("{{\"tag\":{},\"plen\":{},\"taps_from\":{},\"prompt\":{}}}\n",
                             serde_json::to_string(tag).unwrap(), plen, self.taps_from,
                             serde_json::to_string(prompt).unwrap());
        let _ = self.jsonl.write_all(marker.as_bytes());
        let _ = self.jsonl.flush();
        let _ = dev;
    }

    /// After the job's prefill (the prime sink is filled), copy the prompt's tap rows
    /// (positions 0..plen) into the accumulation — must run AFTER admit/prefill.
    pub fn job_prime(&mut self, plen: usize,
                     prime: Option<&Arc<Df2PrimeSink>>, dev: &Arc<cudarc::driver::CudaDevice>) {
        if let Some(ps) = prime {
            if let Ok(cols) = dev.dtoh_sync_copy(&ps.taps) {
                let n = plen.min(cols.len() / TAP_CONCAT_DIM);
                self.taps_from = 0;
                self.taps_n = n;
                self.taps_max = n;
                self.taps.clear();
                self.taps.reserve(n * TAP_CONCAT_DIM);
                for c in 0..n {
                    for k in 0..TAP_CONCAT_DIM {
                        self.taps.push(cols[c * TAP_CONCAT_DIM + k].to_f32());
                    }
                }
                eprintln!("[df2-dump] job {}: plen={plen} prime taps rows={n}", self.job_tag);
            } else {
                eprintln!("[df2-dump] job {}: prime sink read FAILED (plen={plen})", self.job_tag);
            }
        }
    }

    /// Record the verify's fed-span taps: staging cols [0, n) at positions pos..pos+n
    /// (the staging is bf16 [TAP_CONCAT_DIM, 8] col-major — element (k, m) at m*DIM + k).
    /// Overlapping spans rewrite with bit-identical values (R1: decode-capture ==
    /// verify-capture of the same span) — the union accumulates the full committed history.
    pub fn record_span(&mut self, pos: usize, n: usize, staging: &[half::bf16]) {
        if n == 0 || self.job_tag.is_empty() { return; }
        let need = (pos + n).saturating_sub(self.taps_from + self.taps_n);
        if need > 0 {
            // The accumulation may skip positions if the first verify's span doesn't start at
            // taps_from (it does — the first step's span starts at plen = taps_from); pad any
            // gap with zeros so rows stay position-indexed.
            self.taps.resize((self.taps_n + need) * TAP_CONCAT_DIM, 0.0);
            self.taps_n += need;
        }
        let base = pos.saturating_sub(self.taps_from);
        for m in 0..n {
            let row = base + m;
            for k in 0..TAP_CONCAT_DIM {
                self.taps[row * TAP_CONCAT_DIM + k] = staging[m * TAP_CONCAT_DIM + k].to_f32();
            }
        }
        self.taps_max = self.taps_max.max(pos + n);
    }

    /// Per-layer × per-column checksums of the fed-span staging (the JSONL tap fingerprint).
    pub fn tap_checksums(staging: &[half::bf16]) -> Vec<u64> {
        let mut ck = Vec::with_capacity(TAP_LAYERS.len() * BLOCK);
        for li in 0..TAP_LAYERS.len() {
            for m in 0..BLOCK {
                let off = m * TAP_CONCAT_DIM + li * HIDDEN;
                let words: Vec<u16> = staging[off..off + HIDDEN].iter().map(|b| b.to_bits()).collect();
                ck.push(fnv64(&words));
            }
        }
        ck
    }

    /// Write one DFlash2 step record. `h_final` (when Some and step < RAW_STEPS) goes to
    /// hfinal.bin ([8, 5120] f32 per step, appended).
    pub fn record_ring_rows(&mut self, step: u64, rows_k: &[Vec<f32>], rows_v: &[Vec<f32>]) {
        // per layer: k + v row blobs (rows already selected host-side); appended raw f32.
        if step >= RING_STEPS as u64 { return; }
        for li in 0..rows_k.len() {
            let kb: &[u8] = bytemuck::cast_slice(&rows_k[li]);
            let vb: &[u8] = bytemuck::cast_slice(&rows_v[li]);
            let _ = self.ringrows.write_all(kb);
            let _ = self.ringrows.write_all(vb);
        }
    }
    pub fn record_df2(&mut self, r: &Df2StepRec, h_final: Option<&[f32]>) {
        let line = serde_json::to_string(r).expect("df2 dump record");
        let _ = self.jsonl.write_all(line.as_bytes());
        let _ = self.jsonl.write_all(b"\n");
        if let Some(hf) = h_final {
            if r.step < RAW_STEPS as u64 && hf.len() == BLOCK * HIDDEN {
                let bytes: &[u8] = bytemuck::cast_slice(hf);
                let _ = self.hfinal.write_all(bytes);
            }
        }
        self.n_steps += 1;
    }

    pub fn record_mtp(&mut self, r: &MtpStepRec) {
        let line = serde_json::to_string(r).expect("mtp dump record");
        let _ = self.jsonl.write_all(line.as_bytes());
        let _ = self.jsonl.write_all(b"\n");
        self.n_steps += 1;
    }

    /// End the job: write the accumulated raw taps to `taps.bin` (appended as
    /// `[taps_from, taps_from+taps_n)` rows; the analysis indexes rows by absolute position
    /// via the job markers) and reset for the next job.
    pub fn job_end(&mut self) {
        if self.job_tag.is_empty() { return; }
        let mut f = std::fs::OpenOptions::new()
            .create(true).append(true).open(self.out_dir.join("taps.bin"))
            .expect("open taps.bin");
        let bytes: &[u8] = bytemuck::cast_slice(&self.taps);
        let _ = f.write_all(bytes);
        let _ = f.flush();
        let marker = format!("{{\"job_end\":true,\"tag\":{},\"rows\":{},\"from\":{}}}\n",
                             serde_json::to_string(&self.job_tag).unwrap(),
                             self.taps_n, self.taps_from);
        let _ = self.jsonl.write_all(marker.as_bytes());
        let _ = self.jsonl.flush();
        eprintln!("[df2-dump] job {} end: taps rows [{}..{}) {} steps",
                  self.job_tag, self.taps_from, self.taps_from + self.taps_n, self.n_steps);
        self.job_tag.clear();
    }

    /// Whether a job's records are currently open. Serving-mode markers consult this so a new
    /// admission never splits an open job, and the bench harness's explicit markers stay
    /// authoritative (job_end itself no-ops on a closed job, so a lane finish in bench mode
    /// cannot double-close).
    pub fn job_open(&self) -> bool { !self.job_tag.is_empty() }

    pub fn finish(&mut self) {
        self.job_end();
        let _ = self.jsonl.flush();
        eprintln!("[df2-dump] done: {} steps -> {}", self.n_steps, self.out_dir.display());
    }
}
