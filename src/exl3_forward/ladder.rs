//! Ladder / diagnostic dump helpers: `LadderCtx`, `SpDigest`, `digest_words`, and the FwdModel plain-dump
//! methods. Offline reference probes use these to compare layer residuals. See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// Ladder + forward
// ---------------------------------------------------------------------------

pub struct LadderCtx {
    pub dir: String,
    pub step: usize,
    pub layers: Vec<usize>,
    /// Dump only at this forward step (the comparison position; None = every step).
    pub only_step: Option<usize>,
    /// TP-A: capture into memory instead of files (name, row-0 values as f32) — the xtp probe's
    /// per-layer rank-identity digests and cross-TP comparisons. None = the file dumps above.
    pub mem: Option<std::sync::Mutex<Vec<(String, Vec<f32>)>>>,
    /// TP-B: with `mem`, store a 64-bit DIGEST of the whole captured buffer (two f32 carrying the
    /// hash bits) instead of row-0 values — the prefill rank-identity check (G-T1-b) over every
    /// row of a chunk without holding the chunk's activations.
    pub digest: bool,
    /// TP-SP2 (the G-T1-b redesign for row ownership): with `digest`, ALSO a digest per rank row
    /// block of every own-row buffer class (`sp_row_class`), named `<name>@<block>`; None = today.
    pub sp: Option<SpDigest>,
}

/// TP-SP2: what the row-block digests of one sequence-parallel chunk cover (see LadderCtx.sp).
#[derive(Clone, Debug)]
pub struct SpDigest {
    /// the prefill scratch's row capacity (every f32 row buffer is cap x its row width)
    pub cap: usize,
    /// the chunk's rows
    pub c: usize,
    /// every rank's row block of the chunk (rank order)
    pub blocks: Vec<(usize, usize)>,
    /// Some(b) = digest block b only (the sequence-parallel run: this rank's own block); None = every
    /// block (the replicated reference run of the same chunk)
    pub only: Option<usize>,
    /// fault injection (`--xtp-sp-fault L`, this run and rank only): after layer L's residual update,
    /// flip the lowest mantissa bit of the first element of the first own row of the streams
    pub fault_layer: Option<usize>,
}

/// TP-SP2: the prefill buffer classes that hold only the OWN rows in a sequence-parallel chunk (the
/// streams, the mixer / MoE outputs after a reduce-scatter, the inject coefficients, the final mix);
/// embed, xpre / xpre_mlp (x after the all-gather) and every decode buffer stay whole-buffer classes.
pub fn sp_row_class(name: &str) -> bool {
    matches!(name.trim_end_matches(|c: char| c.is_ascii_digit()), "rmid" | "yatt" | "inj" | "ymlp" | "layer" | "precollapse" | "finalmix")
}

/// TP-B: word-wise FNV-1a-style 64-bit digest (the rank-identity hash of a whole buffer).
pub(crate) fn digest_words(it: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in it { h = (h ^ w).wrapping_mul(0x100_0000_01b3); h ^= h >> 29; }
    h
}

impl LadderCtx {
    /// TP-SP2: push `name@b` digests of the row blocks of `words` (row width `rw` words) — every block
    /// (the reference run) or only the own one (the sequence-parallel run).
    fn push_blocks(&self, mm: &std::sync::Mutex<Vec<(String, Vec<f32>)>>, name: &str, words: &[u64], rw: usize) {
        let Some(sp) = self.sp.as_ref() else { return };
        if !sp_row_class(name) || rw == 0 { return; }
        for (b, &(lo, hi)) in sp.blocks.iter().enumerate() {
            if sp.only.map_or(false, |o| o != b) || hi * rw > words.len() { continue; }
            let d = digest_words(words[lo * rw..hi * rw].iter().copied());
            mm.lock().unwrap().push((format!("{name}@{b}"), vec![f32::from_bits(d as u32), f32::from_bits((d >> 32) as u32)]));
        }
    }
    pub(super) fn dump(&self, dev: &Arc<CudaDevice>, buf: &CudaSlice<f32>, name: &str, m: usize) -> Result<()> {
        let host = dev.dtoh_sync_copy(buf)?;
        if let (Some(mm), true) = (self.mem.as_ref(), self.digest) {
            let d = digest_words(host.iter().map(|v| v.to_bits() as u64));
            mm.lock().unwrap().push((name.to_string(), vec![f32::from_bits(d as u32), f32::from_bits((d >> 32) as u32)]));
            if let Some(sp) = self.sp.as_ref() {
                let w: Vec<u64> = host.iter().map(|v| v.to_bits() as u64).collect();
                self.push_blocks(mm, name, &w, host.len() / sp.cap.max(1));
            }
            return Ok(());
        }
        if let Some(mm) = self.mem.as_ref() {
            let rw = host.len() / m.max(1);
            mm.lock().unwrap().push((name.to_string(), host[..rw].to_vec()));
            return Ok(());
        }
        let path = format!("{}/{}_s{:03}.f32", self.dir, name, self.step);
        let mut f = std::fs::File::create(&path)?;
        let rw = host.len() / m.max(1);
        for v in &host[..rw] {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
    /// Per-layer f16-row dump for the verify-vs-plain diff (same convention:
    /// row 0 to disk as f32).
    pub(super) fn dump_u16(&self, dev: &Arc<CudaDevice>, buf: &CudaSlice<u16>, name: &str, n: usize) -> Result<()> {
        let host = dev.dtoh_sync_copy(buf)?;
        if let (Some(mm), true) = (self.mem.as_ref(), self.digest) {
            let d = digest_words(host[..n].iter().map(|&b| b as u64));
            mm.lock().unwrap().push((name.to_string(), vec![f32::from_bits(d as u32), f32::from_bits((d >> 32) as u32)]));
            if let Some(sp) = self.sp.as_ref() {
                let w: Vec<u64> = host[..n].iter().map(|&b| b as u64).collect();
                self.push_blocks(mm, name, &w, n / sp.c.max(1));
            }
            return Ok(());
        }
        if let Some(mm) = self.mem.as_ref() {
            mm.lock().unwrap().push((name.to_string(), host[..n].iter().map(|&b| f16::from_bits(b).to_f32()).collect()));
            return Ok(());
        }
        let path = format!("{}/{}_s{:03}.f32", self.dir, name, self.step);
        let mut f = std::fs::File::create(&path)?;
        for &b in &host[..n] {
            let v = f16::from_bits(b).to_f32();
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
    pub(super) fn want(&self, li: usize) -> bool {
        self.layers.contains(&li) && self.active()
    }
    pub(super) fn want_steps(&self, li: usize) -> bool {
        self.layers.contains(&li) && !self.dir.is_empty()
    }
    pub(super) fn active(&self) -> bool {
        self.only_step.map_or(true, |s| s == self.step)
    }
}

impl FwdModel {
    /// S-A3-f-d Item 1b: plain-path col-0 dump pass (--mtp-dump=<dir>). Runs
    /// AFTER a completed forward_step (never inside a graph capture): reads the
    /// committed col-0 planes of the same buffers the verify side dumps, with a
    /// `_p` name suffix so the two sides never collide in the dump dir. No
    /// kernel launches — pure dtoh + file IO.
    pub(super) fn diag_plain_dump(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        let _ = l;
        let d = std::path::PathBuf::from(crate::opts::var(crate::opt!("mtp-dump"))
            .map_err(|_| anyhow::anyhow!(crate::opt!("mtp-dump")))?);
        std::fs::create_dir_all(&d)?;
        let dump_f = |d: &std::path::PathBuf, buf: &CudaSlice<f32>, name: &str, r: usize| -> Result<()> {
            let host = self.dtoh(buf)?;
            let path = format!("{}/{}_p.f32", d.display(), name);
            let w = buf.len() / m;
            let mut f = std::fs::File::create(&path)?;
            let s = (r * w).min(host.len());
            let e = ((r + 1) * w).min(host.len());
            for &v in &host[s..e] { f.write_all(&v.to_le_bytes())?; }
            Ok(())
        };
        let dump_u = |d: &std::path::PathBuf, buf: &CudaSlice<u16>, name: &str, r: usize| -> Result<()> {
            let host = self.dtoh(buf)?;
            let path = format!("{}/{}_p.f32", d.display(), name);
            let w = buf.len() / m;
            let mut f = std::fs::File::create(&path)?;
            let s = (r * w).min(host.len());
            let e = ((r + 1) * w).min(host.len());
            for &b in &host[s..e] {
                f.write_all(&f16::from_bits(b).to_f32().to_le_bytes())?;
            }
            Ok(())
        };
        // ("stagingCTX"/"finalmix" are dumped by forward_step_kernels at the SAME
        // points as verify_kernels; the post-step reads here use distinct names so
        // they never overwrite the 1:1 comparable files.)
        dump_f(&d, &sc.resid, "endctx", 0)?;
        dump_f(&d, &sc.cos, "stagingCOS", 0)?;
        dump_u(&d, &sc.mixer_out, "rmidfinal", 0)?;
        dump_u(&d, &sc.x, "lmheadin", 0)?;
        dump_u(&d, &sc.logits, "press0logits", 0)?;
        let amh = self.dtoh(&sc.argmax)?;
        crate::dprintln!("PLAIN_DIAG argmax[:{}]= {:?}  v={}", m, &amh[..m], self.cfg.vocab_size);
        Ok(())
    }

    /// EXL27B: plain residual (hc_lowrank == 0): the normed stream IS the mixer input — hn -> x.
    pub(super) fn plain_publish(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        let n = (self.cfg.hidden_size * self.cfg.hc_count.max(1) * m) as i64;
        xqlaunch!(l, "xq_copy_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.x, &sc.hn, n, 0i64, 0i64))?;
        Ok(())
    }

}
