//! WP25 (PLAN/SURPASS_PLAN_2026-09-26.md, owner decision 3): the KV-format quality instrument and
//! the q8 host reference. A child module of `exl3_forward` (declared there with `#[path]`) so the
//! teacher-forced verify step can reach the private verify/prefill internals without widening them.
//!
//! * `q8_quant_row` / `q8_dequant_row` / `h32r`: the reference implementation's "-cq 8" cache format (exllamav3
//!   `cache/q_cache_kernels.cuh` quant_block_x4 + `triton_paged._qc_load_kt/_qc_load_v`, fork
//!   523ecd3) with the kernels' exact op order — the device bytes are reproducible bit-for-bit.
//! * `--probe-exl3-kvq`: every KV format through the SERVED device helpers (xq_kv_put / xq_kv_ld /
//!   xq_kv_ld8 / xq_kv_rot) on synthetic rows, no model: q8 bytes, scales and reads vs the host
//!   reference bitwise, ld8 == ld, and each format's reconstruction SNR.
//! * `--probe-exl3-kvnll`: one arm (= the process's KV format) of the WP25 rung-1 gate. Prefills a
//!   corpus through the served C-token chunk path, teacher-forces `tf` tokens at each window in
//!   6-row verify-class steps (the served verify kernels, forced full commit), scores logprob(true),
//!   argmax and top-5 per token, and — against the f32 arm's dumped logits — ΔNLL, top-1 agreement,
//!   KL and top-5 overlap; then multi-key needles with near-duplicate distractors at the end of the
//!   context and MTP acceptance over 3 prompts. `scripts/kvnll_arms.sh` runs all arms.

use super::*;
use std::io::{BufReader, BufWriter, Read};

// ---------------------------------------------------------------------------------------------
// q8 host reference (bit-exact twin of kernels/exl3_bench.cu xq_kv_put_q8 / xq_kv_ld / xq_h32r)
// ---------------------------------------------------------------------------------------------

/// 1/sqrt(32) as the reference implementation writes it (`r32`).
pub(crate) const R32: f32 = 0.176_776_695_296_636_881_10_f32;

/// Unnormalized H32 butterfly on each 32-group, strides 1,2,4,8,16 — low index a+b, high a-b.
fn h32_raw(v: &mut [f32]) {
    for g in v.chunks_exact_mut(32) {
        let mut o = 1;
        while o < 32 {
            let prev: [f32; 32] = g.try_into().unwrap();
            for j in 0..32 {
                g[j] = if j & o != 0 { prev[j ^ o] - prev[j] } else { prev[j] + prev[j ^ o] };
            }
            o <<= 1;
        }
    }
}

/// Orthonormal H32/sqrt(32) per 32-group (involutory) — the device `xq_h32r`.
pub(crate) fn h32r(v: &mut [f32]) {
    h32_raw(v);
    for x in v.iter_mut() { *x *= R32; }
}

/// One K/V row -> the cache row bytes of `fmt` (KV_Q8 or KV_Q4X): hd codes, then hd/32 f16
/// group scales (LE), zero padding to `kv_rowbytes`.
pub(crate) fn q8_quant_row(x: &[f32], fmt: u32) -> Vec<u8> {
    let hd = x.len();
    assert!(hd % 32 == 0, "q8 rows need hd % 32 == 0");
    let mut row = vec![0u8; kv_rowbytes(fmt, hd)];
    let mut v = x.to_vec();
    h32r(&mut v);
    for (g, grp) in v.chunks_exact(32).enumerate() {
        let s = grp.iter().fold(0.0f32, |m, &a| m.max(a.abs())) + 1e-10f32;
        let inv_s = 1.0f32 / s;
        for (j, &a) in grp.iter().enumerate() {
            let vs = a * inv_s;
            let q = if fmt == KV_Q8 {
                (vs.mul_add(128.0, 128.0).floor() as i32).clamp(0, 255)
            } else {
                (vs.mul_add(8.0, 8.0).floor() as i32).clamp(0, 15) * 16 + 8
            };
            row[g * 32 + j] = q as u8;
        }
        let sb = f16::from_f32(s).to_bits().to_le_bytes();
        row[hd + 2 * g] = sb[0];
        row[hd + 2 * g + 1] = sb[1];
    }
    row
}

/// Cache row -> values in the ROTATED basis: (code - 127.5) * (f16 scale / 128), one rounding.
pub(crate) fn q8_dequant_row(row: &[u8], hd: usize) -> Vec<f32> {
    (0..hd).map(|d| {
        let g = d / 32;
        let s = f16::from_bits(u16::from_le_bytes([row[hd + 2 * g], row[hd + 2 * g + 1]])).to_f32();
        let sm = s * 0.007_812_5;
        (row[d] as f32 - 128.0).mul_add(sm, 0.5 * sm)
    }).collect()
}

// ---------------------------------------------------------------------------------------------
// --probe-exl3-kvq: device helpers vs host reference, no model
// ---------------------------------------------------------------------------------------------

/// Deterministic test rows: Gaussian-ish body, a few heavy outlier dims (the K-cache shape that
/// motivates the rotation), one all-zero row, one constant row, one tiny-magnitude row.
fn kvq_rows(n: usize, hd: usize) -> Vec<f32> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let mut x = Vec::with_capacity(n * hd);
    for r in 0..n {
        for d in 0..hd {
            // sum of 4 uniforms ~ Gaussian; outliers on dims 3, 77, 190 (x 12)
            let g = (rnd() + rnd() + rnd() + rnd() - 2.0) * 1.7;
            let v = match r {
                0 => 0.0,
                1 => 0.75,
                2 => g * 1e-4,
                _ => if d == 3 || d == 77 || d == 190 { g * 12.0 } else { g },
            };
            x.push(v);
        }
    }
    x
}

fn snr_db(x: &[f32], y: &[f32]) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (a, b) in x.iter().zip(y) {
        sig += (*a as f64) * (*a as f64);
        err += (*a as f64 - *b as f64).powi(2);
    }
    if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() }
}

pub fn probe_kv_q8_selftest() -> Result<()> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::exl3_bench::load_module_pub(&dev)?;
    let f = dev.get_func(MODULE, "xq_kv_selftest").context("xq_kv_selftest missing from exl3_bench")?;
    let (n, hd) = (256usize, 256usize);
    let x = kvq_rows(n, hd);
    let x_d = dev.htod_sync_copy(&x)?;
    let mut all_ok = true;
    for fmt in [KV_F32, KV_F16, KV_FP8, KV_Q8, KV_Q4X] {
        let rb = kv_rowbytes(fmt, hd);
        let mut rows_d = dev.alloc_zeros::<u8>(n * rb)?;
        let mut deq_d = dev.alloc_zeros::<f32>(n * hd)?;
        let mut deq8_d = dev.alloc_zeros::<f32>(n * hd)?;
        let mut rec_d = dev.alloc_zeros::<f32>(n * hd)?;
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (hd as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            f.clone().launch(cfg, (&mut rows_d, &mut deq_d, &mut deq8_d, &mut rec_d, &x_d, hd as i32, fmt as i32))
        }.context("launch xq_kv_selftest")?;
        dev.synchronize()?;
        let rows: Vec<u8> = dev.dtoh_sync_copy(&rows_d)?;
        let deq: Vec<f32> = dev.dtoh_sync_copy(&deq_d)?;
        let deq8: Vec<f32> = dev.dtoh_sync_copy(&deq8_d)?;
        let rec: Vec<f32> = dev.dtoh_sync_copy(&rec_d)?;
        let bits_ne = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let ld8_ne = bits_ne(&deq, &deq8);
        let mut ok = ld8_ne == 0;
        let mut extra = String::new();
        match fmt {
            KV_F32 => {
                let ne = bits_ne(&rec, &x);
                ok &= ne == 0;
                extra = format!("rec==x mismatches {ne}");
            }
            KV_F16 => {
                let xr: Vec<f32> = x.iter().map(|&a| f16::from_f32(a).to_f32()).collect();
                let ne = bits_ne(&rec, &xr);
                ok &= ne == 0;
                extra = format!("rec==f16(x) mismatches {ne}");
            }
            KV_Q8 | KV_Q4X => {
                let (mut byte_ne, mut deq_ne, mut rec_ne) = (0usize, 0usize, 0usize);
                for r in 0..n {
                    let xr = &x[r * hd..(r + 1) * hd];
                    let href = q8_quant_row(xr, fmt);
                    let drow = &rows[r * rb..(r + 1) * rb];
                    byte_ne += href[..hd + hd / 16].iter().zip(&drow[..hd + hd / 16]).filter(|(a, b)| a != b).count();
                    let hdq = q8_dequant_row(drow, hd);
                    deq_ne += bits_ne(&hdq, &deq[r * hd..(r + 1) * hd]);
                    let mut hrec = hdq.clone();
                    h32r(&mut hrec);
                    rec_ne += bits_ne(&hrec, &rec[r * hd..(r + 1) * hd]);
                }
                ok &= byte_ne == 0 && deq_ne == 0 && rec_ne == 0;
                // the zero row must read back exactly zero (scale f16(1e-10) underflows to 0)
                let zero_ok = rec[..hd].iter().all(|&v| v == 0.0);
                ok &= zero_ok;
                extra = format!("host-ref mismatches: bytes {byte_ne} deq {deq_ne} rec {rec_ne}; zero-row exact {zero_ok}");
            }
            _ => {}
        }
        // reconstruction SNR over the Gaussian+outlier rows (rows >= 3) and the tiny row
        let body = snr_db(&x[3 * hd..], &rec[3 * hd..]);
        let tiny = snr_db(&x[2 * hd..3 * hd], &rec[2 * hd..3 * hd]);
        println!("KVQ_SELFTEST fmt={:<4} rowbytes={rb:<4} ld8==ld mismatches {ld8_ne}; {extra}; SNR body {body:.2} dB tiny-row {tiny:.2} dB -> {}",
                 kv_fmt_name(fmt), if ok { "PASS" } else { "FAIL" });
        all_ok &= ok;
    }
    if all_ok {
        println!("KVQ_SELFTEST RESULT: PASS (device helpers == host reference; ld8 == ld on every format)");
        Ok(())
    } else {
        println!("KVQ_SELFTEST RESULT: FAIL");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------------------------
// --probe-exl3-kvnll
// ---------------------------------------------------------------------------------------------

pub struct KvNllArgs {
    pub dir: String,
    pub max_pos: usize,
    /// served prefill chunk width C
    pub chunk: usize,
    /// corpus: raw text (tokenized here) or token ids (JSON array / whitespace)
    pub text: Option<String>,
    pub ids: Option<String>,
    /// teacher-forcing start positions (sorted, each TF span must end before the next window)
    pub windows: Vec<usize>,
    /// teacher-forced tokens per window (rounded up to a multiple of the verify width)
    pub tf: usize,
    /// write this arm's logits rows (the f32 reference arm)
    pub dump_ref: Option<String>,
    /// compare against a dumped reference
    pub ref_path: Option<String>,
    /// per-token CSV
    pub out_csv: Option<String>,
    pub needles: bool,
    /// greedy MTP tokens per acceptance prompt (0 = skip)
    pub accept_new: usize,
    /// corpus tokens prefixed to each acceptance prompt
    pub accept_ctx: usize,
    pub label: String,
}

const REF_MAGIC: &[u8; 8] = b"KVNLLREF";

struct RefHeader { vocab: u32, tf: u32, windows: Vec<u32>, corpus_hash: u64 }

impl RefHeader {
    fn bytes(&self) -> Vec<u8> {
        let mut b = REF_MAGIC.to_vec();
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&self.vocab.to_le_bytes());
        b.extend_from_slice(&self.tf.to_le_bytes());
        b.extend_from_slice(&(self.windows.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.corpus_hash.to_le_bytes());
        for w in &self.windows { b.extend_from_slice(&w.to_le_bytes()); }
        b
    }
}

fn fnv1a(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in ids {
        for b in t.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    }
    h
}

/// Per-row log-softmax summary of f16 logits.
struct RowScore { lse: f64, argmax: u32, top5: [u32; 5], nonfinite: bool }

fn f16_lut() -> Vec<f32> { (0..=u16::MAX).map(|b| f16::from_bits(b).to_f32()).collect() }

fn score_row(row: &[u16], lut: &[f32]) -> RowScore {
    let mut mx = f32::NEG_INFINITY;
    let mut am = 0u32;
    let mut nonfinite = false;
    let mut top: [(f32, u32); 5] = [(f32::NEG_INFINITY, u32::MAX); 5];
    for (i, &b) in row.iter().enumerate() {
        let v = lut[b as usize];
        if !v.is_finite() { nonfinite = true; continue; }
        if v > mx { mx = v; am = i as u32; }
        if v > top[4].0 {
            let mut k = 4;
            while k > 0 && v > top[k - 1].0 { top[k] = top[k - 1]; k -= 1; }
            top[k] = (v, i as u32);
        }
    }
    let mut se = 0.0f64;
    for &b in row {
        let v = lut[b as usize];
        if v.is_finite() { se += ((v - mx) as f64).exp(); }
    }
    RowScore { lse: mx as f64 + se.ln(), argmax: am, top5: top.map(|t| t.1), nonfinite }
}

/// KL(ref || arm) in nats over the full vocabulary (non-finite logits skipped on both sides).
fn kl_rows(r: &[u16], rs: &RowScore, a: &[u16], as_: &RowScore, lut: &[f32]) -> f64 {
    let mut kl = 0.0f64;
    for (&rb, &ab) in r.iter().zip(a) {
        let (lr, la) = (lut[rb as usize], lut[ab as usize]);
        if !lr.is_finite() || !la.is_finite() { continue; }
        let lpr = lr as f64 - rs.lse;
        let lpa = la as f64 - as_.lse;
        kl += lpr.exp() * (lpr - lpa);
    }
    kl.max(0.0)
}

#[derive(Default)]
struct WinStats {
    n: usize,
    nll: f64,
    top1: usize,
    nonfinite: usize,
    // vs reference
    nref: usize,
    dnll: Vec<f64>,
    agree: usize,
    kl: Vec<f64>,
    top5_ovl: f64,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() { return 0.0; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

impl WinStats {
    fn line(&mut self, label: &str, fmt: &str, w: usize) -> String {
        let n = self.n.max(1) as f64;
        let mut s = format!("KVNLL window={w} arm={label} fmt={fmt} n={} nll={:.5} top1_true={:.4} nonfinite={}",
                            self.n, self.nll / n, self.top1 as f64 / n, self.nonfinite);
        if self.nref > 0 {
            let m = self.nref as f64;
            let dmean = self.dnll.iter().sum::<f64>() / m;
            let var = self.dnll.iter().map(|d| (d - dmean).powi(2)).sum::<f64>() / (m - 1.0).max(1.0);
            let klm = self.kl.iter().sum::<f64>() / m;
            let (p99, kmax) = (pct(&mut self.kl, 0.99), self.kl.iter().cloned().fold(0.0, f64::max));
            s += &format!(" | vs_ref dNLL={dmean:+.6} se={:.6} top1_agree={:.4} kl_mean={klm:.6} kl_p99={p99:.5} kl_max={kmax:.4} top5_overlap={:.4}",
                          (var / m).sqrt(), self.agree as f64 / m, self.top5_ovl / m);
        }
        s
    }
}

fn read_ids(p: &str) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(p).with_context(|| format!("read {p}"))?;
    serde_json::from_str(&s).or_else(|_| {
        s.split_whitespace().map(|x| x.parse::<u32>()).collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::from)
    })
}

/// A needle pair: the asked key and its near-duplicate distractor (one character apart).
struct Needle { key: String, val: String, dkey: String, dval: String }

fn needles() -> Vec<Needle> {
    let words = ["ORCHID", "TUNDRA", "MAPLE", "COBALT"];
    let mut s: u32 = 0x2545_F491;
    let mut six = || { s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); format!("{}", 100_000 + (s >> 8) % 900_000) };
    words.iter().map(|w| Needle { key: format!("{w}-7"), val: six(), dkey: format!("{w}-1"), dval: six() }).collect()
}

fn needle_text(key: &str, val: &str) -> String {
    format!("\n\nMemo: the access code for vault {key} is {val}.\n\n")
}

fn needle_question(key: &str) -> String {
    format!("\n\nQuestion: What is the access code for vault {key}? Reply with the number only.\nAnswer: The access code for vault {key} is")
}

/// The answer's code: the first 6-digit run in `s` (a repeated key like "ORCHID-7" is skipped),
/// else the first run of digits.
fn first_number(s: &str) -> Option<String> {
    let runs: Vec<&str> = s.split(|c: char| !c.is_ascii_digit()).filter(|r| !r.is_empty()).collect();
    runs.iter().find(|r| r.len() == 6).or(runs.first()).map(|r| r.to_string())
}

/// Splice needles (depth fractions of `span`) into the corpus id stream; later insertions first
/// so earlier positions stay valid. Needles at 10/35/60/85%, their distractors at 22/47/72/93%.
fn splice_needles(ids: &mut Vec<u32>, tok: &crate::tokenizer::QwenTokenizer, span: usize,
                  nd: &[Needle]) -> Result<()> {
    let mut ins: Vec<(usize, Vec<u32>)> = Vec::new();
    for (k, n) in nd.iter().enumerate() {
        let fk = [0.10, 0.35, 0.60, 0.85][k % 4];
        let fd = [0.22, 0.47, 0.72, 0.93][k % 4];
        ins.push(((span as f64 * fk) as usize, tok.encode(&needle_text(&n.key, &n.val), false)?));
        ins.push(((span as f64 * fd) as usize, tok.encode(&needle_text(&n.dkey, &n.dval), false)?));
    }
    ins.sort_by(|a, b| b.0.cmp(&a.0));
    for (at, t) in ins {
        let at = at.min(ids.len());
        ids.splice(at..at, t);
    }
    Ok(())
}

const ACCEPT_PROMPTS: [(&str, &str); 3] = [
    ("code", "def quicksort(arr):\n    \"\"\"Sort a list of integers in ascending order with quicksort.\"\"\"\n"),
    ("prose", "The history of the printing press is the story of how ideas learned to travel. It begins"),
    ("reason", "Q: A train leaves at 9:40 and arrives at 13:05. How long is the trip?\nA: Let's work it out step by step."),
];

impl FwdModel {
    /// WP25: one teacher-forced verify-class step — `toks` (2..=MTP_MAX_K+1 rows of ONE slot) at
    /// positions p..p+m-1 through the SERVED verify kernels (shadow -> verify_kernels), then a
    /// forced full commit (acc2 = {slot, m-1}): every row's GDN/conv/PLE state is kept, and the
    /// K/V, indexer keys and pooled planes the verify wrote for p..p+m-1 stay valid. Returns the
    /// m f16 logits rows. Eager (probe-only; no graph is captured). The host PLE ring needs no
    /// restore: ple_build_embed advanced it by all m rows, and all m commit.
    fn tf_verify_step(&self, sc: &mut Scratch, slot: usize, p: usize, toks: &[i32]) -> Result<Vec<u16>> {
        let m = toks.len();
        anyhow::ensure!(m >= 2 && m <= MTP_MAX_K + 1 && sc.toks.len() >= m, "tf_verify_step: width {m}");
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let mut tv = toks.to_vec();
        tv.resize(sc.toks.len(), 0);
        self.dev.htod_copy_into(tv, &mut sc.toks)?;
        let mut pv: Vec<i32> = (p..p + m).map(|x| x as i32).collect();
        pv.resize(sc.pos.len(), 0);
        self.dev.htod_copy_into(pv, &mut sc.pos)?;
        let mut slotpos = vec![0i32; sc.slots.len()];
        for r in 0..m {
            slotpos[r * 2] = slot as i32;
            slotpos[r * 2 + 1] = (p + r) as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
        let mut emb = vec![0u16; sc.ple_emb.len()];
        self.ple_build_embed(m, toks, &vec![slot; m], &mut emb)?;
        self.dev.htod_copy_into(emb, &mut sc.ple_emb)?;
        let qsa = self.qsa_live(p + m - 1);
        let (qg, _) = self.qsa_grid_bucket(p + m - 1);
        sc.qsa_grid_nblk = qg;
        self.verify_shadow(&l, sc, slot)?;
        self.verify_kernels(&l, sc, m, qsa, slot, 0)?;
        self.dev.htod_copy_into(vec![slot as i32, (m - 1) as i32], &mut sc.acc2)?;
        self.verify_commit(&l, sc)?;
        self.dev.synchronize()?;
        let lg = self.dev.dtoh_sync_copy(&sc.logits)?;
        Ok(lg[..m * self.cfg.vocab_size].to_vec())
    }
}

fn prefill_span(model: &FwdModel, psc: &mut PrefillScratch, ids: &[u32], from: usize, to: usize) -> Result<()> {
    let c = psc.c;
    let mut pos = from;
    while pos < to {
        let end = (pos + c).min(to);
        let toks: Vec<i32> = ids[pos..end].iter().map(|&t| t as i32).collect();
        model.prefill_chunk(psc, &toks, pos, 0, false, None)?;
        pos = end;
    }
    Ok(())
}

pub fn probe_kv_nll(a: &KvNllArgs) -> Result<()> {
    let vw = MTP_MAX_K + 1; // verify-class width (6 rows)
    let tf = a.tf.max(vw).div_ceil(vw) * vw;
    let mut windows = a.windows.clone();
    windows.sort_unstable();
    windows.dedup();
    anyhow::ensure!(!windows.is_empty(), "--kvnll-windows is empty");
    for w in windows.windows(2) {
        anyhow::ensure!(w[0] + tf <= w[1], "window {} + tf {tf} overlaps window {}", w[0], w[1]);
    }
    let wmax = *windows.last().unwrap();
    let end = wmax + tf;
    // needle questions + answers + the acceptance prompts' context all live past `end`
    anyhow::ensure!(a.max_pos >= end + 256 && a.max_pos >= a.accept_ctx + 1024,
                    "--max-seq-len {} too small for windows up to {wmax} + tf {tf}", a.max_pos);
    // snapshots for the needle questions (restore + suffix prefill = the serve's prefix-cache path)
    crate::opts::set(crate::opt!("exl3-prefix"), "1");

    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir.trim_end_matches('/')))?;
    let mut ids: Vec<u32> = match (&a.ids, &a.text) {
        (Some(p), _) => read_ids(p)?,
        (None, Some(p)) => tok.encode(&std::fs::read_to_string(p).with_context(|| format!("read {p}"))?, false)?,
        (None, None) => bail!("--probe-exl3-kvnll needs --kvnll-text <file> or --kvnll-ids <file>"),
    };
    let nd = if a.needles { needles() } else { Vec::new() };
    if a.needles { splice_needles(&mut ids, &tok, wmax, &nd)?; }
    anyhow::ensure!(ids.len() > end, "corpus has {} tokens; windows need > {end}", ids.len());
    let used = &ids[..=end];
    let corpus_hash = fnv1a(used);

    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    let fmt = model.kv_fmt;
    let fname = kv_fmt_name(fmt);
    let (nkv, hd) = (model.cfg.num_kv_heads, model.cfg.head_dim);
    let kv_gb = (model.k_cache.len() + model.mtp.is_some() as usize) as f64
        * (2 * nkv * a.max_pos * kv_rowbytes(fmt, hd)) as f64 / 1e9;
    println!("KVNLL arm={} fmt={fname} rowbytes={} kv_cache={kv_gb:.2} GB (1 slot x {} pos) corpus={} tok hash={corpus_hash:016x} windows={windows:?} tf={tf} C={} needles={} qsa_splits_dec={}",
             a.label, kv_rowbytes(fmt, hd), a.max_pos, ids.len(), a.chunk, nd.len(), qsa_splits_dec());
    let v = model.cfg.vocab_size;
    anyhow::ensure!(used.iter().all(|&t| (t as usize) < v), "corpus has token ids >= vocab {v}");
    let mut sc = Scratch::new(&model.dev, &model.cfg, vw, model.cfg.rotary_dim)?;
    let mut psc_opt: Option<PrefillScratch> = Some(model.prefill_scratch(a.chunk)?);
    model.reset_slot(0)?;

    let hdr = RefHeader { vocab: v as u32, tf: tf as u32, windows: windows.iter().map(|&w| w as u32).collect(), corpus_hash };
    let mut refw = match &a.dump_ref {
        Some(p) => {
            let mut w = BufWriter::with_capacity(1 << 24, std::fs::File::create(p).with_context(|| format!("create {p}"))?);
            w.write_all(&hdr.bytes())?;
            Some(w)
        }
        None => None,
    };
    let mut refr = match &a.ref_path {
        Some(p) => {
            let mut r = BufReader::with_capacity(1 << 24, std::fs::File::open(p).with_context(|| format!("open {p}"))?);
            let want = hdr.bytes();
            let mut got = vec![0u8; want.len()];
            r.read_exact(&mut got)?;
            anyhow::ensure!(got == want, "reference {p} was dumped with a different corpus/windows/tf/vocab (header mismatch)");
            Some(r)
        }
        None => None,
    };
    let mut csv = match &a.out_csv {
        Some(p) => {
            let mut w = BufWriter::new(std::fs::File::create(p)?);
            writeln!(w, "label,fmt,window,pos,true,logp_true,argmax,top5,ref_logp_true,ref_argmax,kl,top5_overlap")?;
            Some(w)
        }
        None => None,
    };
    let lut = f16_lut();
    let mut rrow = vec![0u16; v];
    let mut rbytes = vec![0u8; v * 2];
    let mut pos = 0usize;
    let mut results: Vec<String> = Vec::new();
    let t_all = std::time::Instant::now();
    for &w in &windows {
        let tp = std::time::Instant::now();
        prefill_span(&model, psc_opt.as_mut().unwrap(), &ids, pos, w)?;
        model.dev.synchronize()?;
        let pf_s = tp.elapsed().as_secs_f64();
        let tt = std::time::Instant::now();
        let mut st = WinStats::default();
        let mut p = w;
        while p < w + tf {
            let toks: Vec<i32> = ids[p..p + vw].iter().map(|&t| t as i32).collect();
            let lg = model.tf_verify_step(&mut sc, 0, p, &toks)?;
            for r in 0..vw {
                let row = &lg[r * v..(r + 1) * v];
                let truth = ids[p + r + 1];
                let rs = score_row(row, &lut);
                let lpt = lut[row[truth as usize] as usize] as f64 - rs.lse;
                st.n += 1;
                st.nll -= lpt;
                st.top1 += (rs.argmax == truth) as usize;
                st.nonfinite += rs.nonfinite as usize;
                if let Some(wr) = refw.as_mut() {
                    for &b in row { wr.write_all(&b.to_le_bytes())?; }
                }
                let mut cmp = (f64::NAN, u32::MAX, f64::NAN, f64::NAN);
                if let Some(rd) = refr.as_mut() {
                    rd.read_exact(&mut rbytes).context("reference file truncated")?;
                    for (i, c) in rbytes.chunks_exact(2).enumerate() { rrow[i] = u16::from_le_bytes([c[0], c[1]]); }
                    let rsr = score_row(&rrow, &lut);
                    let lpr = lut[rrow[truth as usize] as usize] as f64 - rsr.lse;
                    let kl = kl_rows(&rrow, &rsr, row, &rs, &lut);
                    let ovl = rs.top5.iter().filter(|t| rsr.top5.contains(t)).count() as f64 / 5.0;
                    st.nref += 1;
                    st.dnll.push(lpr - lpt); // NLL_arm - NLL_ref
                    st.agree += (rs.argmax == rsr.argmax) as usize;
                    st.kl.push(kl);
                    st.top5_ovl += ovl;
                    cmp = (lpr, rsr.argmax, kl, ovl);
                }
                if let Some(c) = csv.as_mut() {
                    let t5: Vec<String> = rs.top5.iter().map(|x| x.to_string()).collect();
                    writeln!(c, "{},{fname},{w},{},{truth},{lpt:.6},{},{},{:.6},{},{:.8},{:.2}",
                             a.label, p + r, rs.argmax, t5.join(" "), cmp.0,
                             if cmp.1 == u32::MAX { -1 } else { cmp.1 as i64 }, cmp.2, cmp.3)?;
                }
            }
            p += vw;
        }
        let line = st.line(&a.label, fname, w);
        println!("{line} | prefill {:.1} s ({:.0} tok/s) tf {:.1} s ({:.1} ms/step)",
                 pf_s, (w - pos) as f64 / pf_s.max(1e-9), tt.elapsed().as_secs_f64(),
                 tt.elapsed().as_secs_f64() * 1e3 / (tf / vw) as f64);
        results.push(line);
        pos = w + tf;
    }
    if let Some(mut wr) = refw.take() { wr.flush()?; }
    if let Some(mut c) = csv.take() { c.flush()?; }

    // ---- needles: questions at the end of the whole context (restore + suffix, the serve's
    // prefix-cache path; the seam token through forward_step), greedy 10 tokens each.
    let mut npass = 0usize;
    if !nd.is_empty() {
        model.snapshot_slot(0, psc_opt.as_ref().unwrap())?;
        for n in &nd {
            model.restore_slot(0, psc_opt.as_mut().unwrap())?;
            let q = tok.encode(&needle_question(&n.key), false)?;
            let ql = q.len();
            anyhow::ensure!(ql >= 2, "needle question tokenized to {ql} tokens");
            // question[..ql-1] as chunked prefill from `end`, the last token as the seam decode step
            {
                let psc = psc_opt.as_mut().unwrap();
                let body: Vec<i32> = q[..ql - 1].iter().map(|&t| t as i32).collect();
                let mut off = 0usize;
                while off < body.len() {
                    let e = (off + psc.c).min(body.len());
                    model.prefill_chunk(psc, &body[off..e], end + off, 0, false, None)?;
                    off = e;
                }
            }
            let mut gen: Vec<u32> = Vec::new();
            let mut next = model.forward_step(&mut sc, &[q[ql - 1] as i32], &[end + ql - 1], &[0], None)?[0];
            for i in 0..10 {
                gen.push(next as u32);
                if i == 9 { break; }
                next = model.forward_step(&mut sc, &[next], &[end + ql + i], &[0], None)?[0];
            }
            let text = tok.decode(&gen, true)?;
            let got = first_number(&text);
            let verdict = match got.as_deref() {
                Some(g) if g == n.val => { npass += 1; "PASS" }
                Some(g) if g == n.dval => "DISTRACTED",
                _ => "FAIL",
            };
            println!("KVNLL_NEEDLE arm={} fmt={fname} ctx={end} key={} expect={} distractor={} got={:?} -> {verdict}",
                     a.label, n.key, n.val, n.dval, text.trim());
        }
        results.push(format!("KVNLL needles arm={} fmt={fname} ctx={end} pass={npass}/{}", a.label, nd.len()));
    }

    // ---- MTP acceptance: fresh slot per prompt, served seam + prime, fixed-depth chain rounds.
    if a.accept_new > 0 {
        let head = model.mtp.as_ref().context("pack has no MTP head: acceptance unavailable")?;
        let depth = mtp_depth_default();
        let (mut tot_acc, mut tot_drafted, mut tot_rounds, mut tot_emit) = (0usize, 0usize, 0usize, 0usize);
        for (name, text) in ACCEPT_PROMPTS {
            model.reset_slot(0)?;
            let mut prompt: Vec<u32> = ids[..a.accept_ctx.min(ids.len())].to_vec();
            prompt.extend(tok.encode(text, false)?);
            let plen = prompt.len();
            model.prefill_prompt(&mut sc, &prompt, Some(a.chunk), &mut psc_opt)?;
            let b0 = model.dev.dtoh_sync_copy(&sc.argmax)?[0];
            // the served seam prime (WP03, merged into p2 after this probe was written)
            model.mtp_seam_prime(&mut sc, psc_opt.as_ref(), head, prompt[plen - 1] as i32, plen, 0)?;
            let mut out_toks: Vec<u32> = vec![b0 as u32];
            let (mut b, mut p, mut tap_row) = (b0, plen, 0usize);
            let (mut acc, mut rounds) = (0usize, 0usize);
            while out_toks.len() < a.accept_new && !EOS_IDS.contains(&b) {
                let o = model.mtp_round(&mut sc, head, 0, depth, b, p, tap_row)?;
                acc += o.a;
                rounds += 1;
                out_toks.extend(o.emitted.iter().map(|&e| e as u32));
                b = o.last_tok;
                p = o.pos_next;
                tap_row = o.tap_row;
            }
            let emitted = out_toks.len() - 1;
            println!("KVNLL_ACCEPT arm={} fmt={fname} prompt={name} ctx={plen} depth={depth} rounds={rounds} accepted={acc}/{} ({:.2}%) emitted/round={:.3} tokens_hash={:016x} head={:?}",
                     a.label, rounds * depth, 100.0 * acc as f64 / (rounds * depth).max(1) as f64,
                     emitted as f64 / rounds.max(1) as f64, fnv1a(&out_toks), &out_toks[..out_toks.len().min(12)]);
            tot_acc += acc;
            tot_drafted += rounds * depth;
            tot_rounds += rounds;
            tot_emit += emitted;
        }
        results.push(format!("KVNLL accept arm={} fmt={fname} accepted={tot_acc}/{tot_drafted} ({:.2}%) emitted/round={:.3}",
                             a.label, 100.0 * tot_acc as f64 / tot_drafted.max(1) as f64,
                             tot_emit as f64 / tot_rounds.max(1) as f64));
    }
    println!("==== KVNLL_RESULT arm={} fmt={fname} ({:.1} s) ====", a.label, t_all.elapsed().as_secs_f64());
    for r in &results { println!("KVNLL_RESULT {r}"); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, hd: usize) -> Vec<f32> { kvq_rows(n, hd) }

    /// A plain e4m3 (satfinite, RNE, subnormals) quantizer with the fp8 cache's row scale —
    /// only to check the "q8 is well above fp8 at the same bytes" simulation claim.
    fn e4m3(x: f32) -> f32 {
        let a = x.abs().min(448.0);
        if a == 0.0 { return 0.0; }
        let e = a.log2().floor().max(-6.0);
        let step = (2.0f32).powf(e - 3.0);
        let q = (a / step).round_ties_even() * step;
        q.min(448.0).copysign(x)
    }

    #[test]
    fn h32r_is_involutory() {
        let mut v = rows(4, 256)[3 * 256..].to_vec();
        let x = v.clone();
        h32r(&mut v);
        h32r(&mut v);
        for (a, b) in v.iter().zip(&x) { assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}"); }
    }

    /// p3c (WP13 x WP25): the dense-v3 accumulate epilogue leaves the q8 rotated basis with a
    /// STAGED H32 butterfly — thread dgl (0..16) of a 16*DPT-dim slice holds dims dgl*DPT + e;
    /// stages st < DPT pair registers e, e^st, stages st >= DPT pair thread dgl with dgl ^ st/DPT
    /// through __shfl_xor (high = partner - self, low = self + partner), then one multiply by
    /// R32. Emulated here op for op (f32 add/sub/mul are single IEEE RN ops, as the kernel's
    /// __fadd_rn/__fsub_rn/__fmul_rn) and required bit-identical to h32r (= the device xq_h32r
    /// that v2 / xq_attn_decode apply per 32-lane warp) for DPT 2 (acc1's q8 body) and 4 (acc4).
    #[test]
    fn dv3_q8_staged_unrotation_is_h32r() {
        let hd = 256usize;
        let x = rows(3, hd);
        for dpt in [2usize, 4] {
            let sd = 16 * dpt;
            for r in 0..3 {
                let xr = &x[r * hd..(r + 1) * hd];
                let mut want = xr.to_vec();
                h32r(&mut want);
                let mut got = vec![0f32; hd];
                for bx in 0..hd / sd {
                    let mut o: Vec<Vec<f32>> =
                        (0..16).map(|dgl| (0..dpt).map(|e| xr[bx * sd + dgl * dpt + e]).collect()).collect();
                    let mut st = 1usize;
                    while st < 32 {
                        let prev = o.clone();
                        if st < dpt {
                            for dgl in 0..16 {
                                for e in 0..dpt {
                                    o[dgl][e] = if e & st != 0 { prev[dgl][e ^ st] - prev[dgl][e] }
                                                else { prev[dgl][e] + prev[dgl][e ^ st] };
                                }
                            }
                        } else {
                            let tm = st / dpt;
                            assert!(tm < 16, "a shuffle partner must stay inside the head's 16 threads");
                            for dgl in 0..16 {
                                let hi = dgl & tm != 0;
                                for e in 0..dpt {
                                    let pv = prev[dgl ^ tm][e];
                                    o[dgl][e] = if hi { pv - prev[dgl][e] } else { prev[dgl][e] + pv };
                                }
                            }
                        }
                        st <<= 1;
                    }
                    for dgl in 0..16 {
                        for e in 0..dpt { got[bx * sd + dgl * dpt + e] = o[dgl][e] * R32; }
                    }
                }
                let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(bad, 0, "DPT {dpt} row {r}: staged butterfly != h32r in {bad} of {hd} dims");
            }
        }
    }

    #[test]
    fn q8_layout_and_rowbytes() {
        assert_eq!(kv_rowbytes(KV_Q8, 256), 272);
        assert_eq!(kv_rowbytes(KV_Q8, 128), 144);
        assert_eq!(kv_rowbytes(KV_Q4X, 256), 272);
        assert_eq!(kv_rowbytes(KV_FP8, 256), 272);
        assert_eq!(kv_rowbytes(KV_F32, 256), 1024);
        let x = rows(8, 256);
        let r = q8_quant_row(&x[5 * 256..6 * 256], KV_Q8);
        assert_eq!(r.len(), 272);
        assert!(r[256 + 16..].iter().all(|&b| b == 0), "padding must stay zero");
        for g in 0..8 {
            let s = f16::from_bits(u16::from_le_bytes([r[256 + 2 * g], r[257 + 2 * g]])).to_f32();
            assert!(s > 0.0 && s.is_finite());
        }
    }

    #[test]
    fn q8_zero_row_reads_zero() {
        let r = q8_quant_row(&[0.0f32; 256], KV_Q8);
        assert!(r[..256].iter().all(|&c| c == 128), "zero maps to the first code above the midpoint");
        let d = q8_dequant_row(&r, 256);
        assert!(d.iter().all(|&v| v == 0.0), "f16(1e-10) underflows: scale 0 -> exact zeros");
    }

    #[test]
    fn q8_error_is_bounded_by_half_a_step() {
        let x = rows(64, 256);
        for r in 3..64 {
            let xr = &x[r * 256..(r + 1) * 256];
            let mut rot = xr.to_vec();
            h32r(&mut rot);
            let row = q8_quant_row(xr, KV_Q8);
            let d = q8_dequant_row(&row, 256);
            for (g, (dg, rg)) in d.chunks(32).zip(rot.chunks(32)).enumerate() {
                let s = f16::from_bits(u16::from_le_bytes([row[256 + 2 * g], row[257 + 2 * g]])).to_f32();
                let smax = rg.iter().fold(0.0f32, |m, &a| m.max(a.abs()));
                for (a, b) in dg.iter().zip(rg) {
                    // half a grid step (s/128) plus the f16 scale rounding (<= 2^-11 relative)
                    assert!((a - b).abs() <= s / 256.0 + smax * 1.0e-3 + 1e-6, "row {r} g {g}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn q8_beats_fp8_and_q4x_is_broken() {
        let x = rows(200, 256);
        let (mut e8, mut e4, mut ef, mut sig) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for r in 3..200 {
            let xr = &x[r * 256..(r + 1) * 256];
            let mut q8 = q8_dequant_row(&q8_quant_row(xr, KV_Q8), 256);
            h32r(&mut q8);
            let mut q4 = q8_dequant_row(&q8_quant_row(xr, KV_Q4X), 256);
            h32r(&mut q4);
            let amax = xr.iter().fold(0.0f32, |m, &a| m.max(a.abs()));
            let sc = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            for d in 0..256 {
                let f8 = e4m3(xr[d] / sc) * sc;
                sig += (xr[d] as f64).powi(2);
                e8 += ((xr[d] - q8[d]) as f64).powi(2);
                e4 += ((xr[d] - q4[d]) as f64).powi(2);
                ef += ((xr[d] - f8) as f64).powi(2);
            }
        }
        let (s8, s4, sf) = (10.0 * (sig / e8).log10(), 10.0 * (sig / e4).log10(), 10.0 * (sig / ef).log10());
        eprintln!("SNR q8 {s8:.2} dB, fp8 {sf:.2} dB, q4x {s4:.2} dB");
        assert!(s8 > sf + 8.0, "q8 {s8:.2} dB should clear fp8 {sf:.2} dB by a wide margin");
        assert!(s8 > 38.0, "q8 SNR {s8:.2}");
        assert!(s4 < s8 - 18.0, "the 4-bit control must be clearly worse ({s4:.2} vs {s8:.2})");
    }

    /// The kernels' fold-out (q rotated once, scores/weights on the ROTATED dequant, output
    /// rotated back once) == dequantize-and-unrotate every K/V row then plain attention, and both
    /// stay close to the exact f32 attention.
    #[test]
    fn rotation_folds_out_of_attention() {
        let (hd, np) = (256usize, 48usize);
        let x = rows(3 + 2 * np + 1, hd);
        let kx = &x[3 * hd..(3 + np) * hd];
        let vx = &x[(3 + np) * hd..(3 + 2 * np) * hd];
        let q: Vec<f32> = x[(3 + 2 * np) * hd..].iter().map(|v| v * 0.3).collect();
        let scale = 1.0 / (hd as f64).sqrt();
        let attend = |q: &[f32], k: &dyn Fn(usize) -> Vec<f32>, v: &dyn Fn(usize) -> Vec<f32>| -> Vec<f64> {
            let s: Vec<f64> = (0..np).map(|t| k(t).iter().zip(q).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale).collect();
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|z| (z - mx).exp()).collect();
            let den: f64 = w.iter().sum();
            let mut o = vec![0.0f64; hd];
            for t in 0..np { for (d, vv) in v(t).iter().enumerate() { o[d] += w[t] * *vv as f64 / den; } }
            o
        };
        let exact = attend(&q, &|t| kx[t * hd..(t + 1) * hd].to_vec(), &|t| vx[t * hd..(t + 1) * hd].to_vec());
        let kq: Vec<Vec<u8>> = (0..np).map(|t| q8_quant_row(&kx[t * hd..(t + 1) * hd], KV_Q8)).collect();
        let vq: Vec<Vec<u8>> = (0..np).map(|t| q8_quant_row(&vx[t * hd..(t + 1) * hd], KV_Q8)).collect();
        // (b) the kernels: rotated q, rotated-basis reads, one rotation of the output
        let mut qr = q.clone();
        h32r(&mut qr);
        let ob = attend(&qr, &|t| q8_dequant_row(&kq[t], hd), &|t| q8_dequant_row(&vq[t], hd));
        let mut ob32: Vec<f32> = ob.iter().map(|&z| z as f32).collect();
        h32r(&mut ob32);
        // (c) dequantize-in-reader: every row back to the model basis, plain attention
        let unrot = |r: &Vec<u8>| { let mut d = q8_dequant_row(r, hd); h32r(&mut d); d };
        let oc = attend(&q, &|t| unrot(&kq[t]), &|t| unrot(&vq[t]));
        let norm = exact.iter().map(|z| z * z).sum::<f64>().sqrt();
        let e_bc = ob32.iter().zip(&oc).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>().sqrt() / norm;
        let e_ex = ob32.iter().zip(&exact).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>().sqrt() / norm;
        eprintln!("fold-out vs dequant-in-reader rel {e_bc:.3e}; q8 attention vs exact rel {e_ex:.3e}");
        assert!(e_bc < 1e-5, "fold-out must equal the per-row unrotation up to f32 rounding ({e_bc})");
        assert!(e_ex < 2e-2, "q8 attention output error {e_ex}");
    }

    #[test]
    fn scoring_helpers() {
        let lut = f16_lut();
        let row: Vec<u16> = [0.5f32, 3.0, -1.0, 2.0, 3.0, 0.0, 1.0].iter().map(|&v| f16::from_f32(v).to_bits()).collect();
        let s = score_row(&row, &lut);
        assert_eq!(s.argmax, 1, "first maximum wins");
        assert_eq!(&s.top5[..3], &[1, 4, 3]);
        let z: f64 = row.iter().map(|&b| (lut[b as usize] as f64 - s.lse).exp()).sum();
        assert!((z - 1.0).abs() < 1e-12);
        assert!(kl_rows(&row, &s, &row, &s, &lut).abs() < 1e-15);
        let row2: Vec<u16> = [0.5f32, 2.0, -1.0, 3.0, 3.0, 0.0, 1.0].iter().map(|&v| f16::from_f32(v).to_bits()).collect();
        let s2 = score_row(&row2, &lut);
        assert!(kl_rows(&row, &s, &row2, &s2, &lut) > 1e-3);
        assert_eq!(first_number(" 482915.\n"), Some("482915".to_string()));
        assert_eq!(first_number(" ORCHID-7 is 482915"), Some("482915".to_string()));
        assert_eq!(first_number(" 12 then"), Some("12".to_string()));
        assert_eq!(first_number("none"), None);
        let nd = needles();
        assert_eq!(nd.len(), 4);
        for n in &nd {
            assert_eq!(n.val.len(), 6);
            assert_ne!(n.val, n.dval);
            assert_eq!(n.key.len(), n.dkey.len());
        }
    }
}
