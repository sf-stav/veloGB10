//! Q1 (pack quality vs path): our EXL3 decode against exllamav3's OWN decode, bitwise, on a sample of a real pack.
//!
//! `--probe-dsv41-q1 --model-dir <EXL3 pack or slice> --dsv41-kernel-fixture <oracle dir>`
//!
//! The oracle dir is written by `tools/dsv41_q1/oracle_gen.py` from exllamav3 1.6.2's `ext.reconstruct` (the pack
//! quantiser's decode, built from its own sources): per sampled tensor the rotated-basis weight W_hat fp16 [in, out].
//! Checked here, per tensor:
//!   * host: `crate::exl3::decode_trellis_host` (the decode every EXL3 kernel family was gated against) == W_hat, bit for bit;
//!   * GPU: band 10's reconstruct kernel (`d41x_dw_recon`, the decode `d41x_dec_group` that the dense decode class 12_*
//!     and the dense wide trellis arm 17_* share) == W_hat, bit for bit;
//!   * per (class, bits) pair: the dense linear on x (40 rows: rows 0..16 through the decode class, all 40 through the
//!     wide class, trellis and reconstruct arms) against y = x . W_full in fp64 with W_full built in exllamav3's
//!     get_weight_tensor order (diag(suh) H128 W_hat H128 diag(svh)): the Hadamard / suh / svh convention;
//!   * per routed-expert triple of the sample: the MoE kernels (decode <= 32 rows, prefill > 32 rows, generic) on one
//!     expert against the f64 chain on exllamav3's W_hat (the MoE decode kernel 15_* has its own word decode).
//! Result line `DSV41_Q1_ORACLE_OK` only when every host and GPU decode is bit-identical and every linear / expert is
//! inside its tolerance.

use super::dense_wide::{dense_wide, scratch_parts, WideSrc};
use super::moe::{self, MoePath};
use super::probe::{dp, f2bf, h2f, ref_expert, Cx, Ex, Lin, Rng};
use crate::dsv41::st::{read_header, read_tensor, StHeader};
use crate::dsv41::weights::{exl3::Exl3Weights, Accum, Act, ActDtype, DenseWeight, DevPtr, Dsv41Weights, Launch, MoeWeight, Route, Scratch, Shard};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(serde::Deserialize, Clone)]
struct LinT { rows: usize, x: String, y: String, fused_vs_formula_max_fp16_ulps: f64 }
#[derive(serde::Deserialize, Clone)]
struct Entry { module: String, cls: String, bits: usize, k: usize, n: usize, blocks: usize, file: String, lin: Option<LinT> }
#[derive(serde::Deserialize)]
struct Manifest { entries: Vec<Entry> }

struct Pack { hdr: HashMap<String, usize>, files: Vec<StHeader> }
impl Pack {
    fn open(dir: &Path) -> Result<Pack> {
        let mut files = Vec::new();
        let mut names: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x == "safetensors")).collect();
        names.sort();
        let mut hdr = HashMap::new();
        for p in names {
            let h = read_header(&p)?;
            for t in &h.tensors { hdr.insert(t.name.clone(), files.len()); }
            files.push(h);
        }
        ensure!(!files.is_empty(), "{}: no safetensors files", dir.display());
        Ok(Pack { hdr, files })
    }
    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let i = *self.hdr.get(name).with_context(|| format!("pack lacks {name}"))?;
        read_tensor(&self.files[i], name)
    }
    fn u16s(&self, name: &str) -> Result<Vec<u16>> { Ok(self.bytes(name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()) }
    fn i16s(&self, name: &str) -> Result<Vec<i16>> { Ok(self.bytes(name)?.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()) }
}

fn read_u16(p: &Path) -> Result<Vec<u16>> { Ok(std::fs::read(p).with_context(|| p.display().to_string())?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()) }
fn read_f32(p: &Path) -> Result<Vec<f32>> { Ok(std::fs::read(p).with_context(|| p.display().to_string())?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) }
fn read_f64(p: &Path) -> Result<Vec<f64>> { Ok(std::fs::read(p).with_context(|| p.display().to_string())?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()) }

fn rel(a: &[f32], b: &[f64]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += (*x as f64 - y).powi(2); d += y * y; }
    (n / d.max(1e-300)).sqrt()
}

/// GPU decode of the whole tensor through the wide path's reconstruct arm: W16T [n][k] fp16 bits.
fn gpu_recon(cx: &Cx, w: &DenseWeight, k: usize, n: usize) -> Result<Vec<u16>> {
    let rows = 33usize;
    let r = super::dense::resolve(w, rows)?;
    let (xb, wb) = scratch_parts(&r, rows, WideSrc::Recon);
    ensure!(wb == ((n * k * 2 + 255) & !255), "{}: reconstruct scratch {wb} B for {n} x {k}", w.key);
    let ds = cx.zeros::<u8>(xb + wb)?;
    let dx = cx.zeros::<f32>(rows * k)?;
    let dy = cx.zeros::<f32>(rows * n)?;
    let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: k, ld: k, dtype: ActDtype::F32 };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: n, ld: n, dtype: ActDtype::F32 };
    dense_wide(&cx.l(), w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&ds)), bytes: xb + wb }, WideSrc::Recon)?;
    let all = cx.dl(&ds)?;
    Ok(all[xb..xb + n * k * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// y [rows, n] of the served dense call (decode class for rows <= 32, wide class above; `src` picks the wide arm).
fn run_dense(cx: &Cx, w: &DenseWeight, x: &[f32], rows: usize, k: usize, n: usize, src: Option<WideSrc>) -> Result<Vec<f32>> {
    let dx = cx.up(&x[..rows * k])?;
    let dy = cx.zeros::<f32>(rows * n)?;
    let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: k, ld: k, dtype: ActDtype::F32 };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: n, ld: n, dtype: ActDtype::F32 };
    match src {
        None => {
            let b = Exl3Weights;
            let sb = b.scratch_bytes(w, rows);
            let ds = cx.zeros::<u8>(sb.max(256))?;
            b.dense(&cx.l(), w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb })?;
        }
        Some(s) => {
            let r = super::dense::resolve(w, rows)?;
            let (xb, wb) = scratch_parts(&r, rows, s);
            let ds = cx.zeros::<u8>(xb + wb)?;
            dense_wide(&cx.l(), w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&ds)), bytes: xb + wb }, s)?;
        }
    }
    cx.dl(&dy)
}

fn one_expert_moe(cx: &Cx, ex: &Ex, x: &[f32], rows: usize, path: MoePath, inter: usize) -> Result<Vec<f32>> {
    moe::clear_tables();
    let dim = 5120usize;
    let mw = MoeWeight { layer: 0, n_experts: 1, local: 0..1, top_k: 1, dim, inter, swiglu_limit: 10.0,
        experts: vec![[ex.l[0].dense("w1", inter, dim, Shard::Whole), ex.l[1].dense("w2", dim, inter, Shard::Whole), ex.l[2].dense("w3", inter, dim, Shard::Whole)]] };
    let sb = moe::scratch_for(&mw, None, rows, path)?;
    let ds = cx.zeros::<u8>(sb.max(256))?;
    let dx = cx.up(&x[..rows * dim])?;
    let ids = vec![0i32; rows];
    let wts = vec![0.75f32; rows];
    let (di, dw, dy) = (cx.up(&ids)?, cx.up(&wts)?, cx.zeros::<f32>(rows * dim)?);
    let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: dim, ld: dim, dtype: ActDtype::F32 };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: dim, ld: dim, dtype: ActDtype::F32 };
    let l = Launch { stream: cx.stream, pdl: false };
    let rt = Route { ids: DevPtr(dp(&di)), weights: DevPtr(dp(&dw)), top_k: 1 };
    moe::moe_ex(&l, &mw, None, &xa, &rt, &ya, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb }, path)?;
    cx.dl(&dy)
}

pub fn run(pack_dir: &Path, oracle: &Path) -> Result<()> {
    let m: Manifest = serde_json::from_str(&std::fs::read_to_string(oracle.join("manifest.json")).context("oracle manifest.json")?)?;
    let pack = Pack::open(pack_dir)?;
    let cx = Cx::new()?;
    super::prepare()?;
    // (class, bits) -> [tensors, blocks, host bad values, gpu bad values]
    let mut per: BTreeMap<(String, usize), [u64; 4]> = BTreeMap::new();
    let mut fails: Vec<String> = Vec::new();
    let mut lin_lines: Vec<String> = Vec::new();
    // expert triples: (layer prefix, expert) -> [w1, w2, w3] entries' exllamav3 weights
    let mut trip: BTreeMap<String, [Option<(Lin, Vec<f32>)>; 3]> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    for (i, e) in m.entries.iter().enumerate() {
        let tr = pack.i16s(&format!("{}.trellis", e.module))?;
        let suh = pack.u16s(&format!("{}.suh", e.module))?;
        let svh = pack.u16s(&format!("{}.svh", e.module))?;
        ensure!(tr.len() == e.k / 16 * e.n / 16 * 16 * e.bits, "{}: trellis length {} for {}x{} K{}", e.module, tr.len(), e.k, e.n, e.bits);
        let refw = read_u16(&oracle.join(&e.file))?;
        ensure!(refw.len() == e.k * e.n, "{}: oracle dump {} values, want {}", e.module, refw.len(), e.k * e.n);
        // host
        let ours = crate::exl3::decode_trellis_host(&tr, e.k / 16, e.n / 16, 2 * e.bits);
        let hbad = ours.iter().zip(&refw).filter(|(a, b)| a != b).count() as u64;
        // GPU reconstruct kernel
        let lin = Lin::from_host(&cx, 2 * e.bits, e.k, e.n, tr, suh, svh, false)?;
        let dw = lin.dense(&e.module, e.n, e.k, Shard::Whole);
        let g = gpu_recon(&cx, &dw, e.k, e.n)?;
        let mut gbad = 0u64;
        for nn in 0..e.n { for kk in 0..e.k { if g[nn * e.k + kk] != refw[kk * e.n + nn] { gbad += 1; } } }
        let s = per.entry((e.cls.clone(), e.bits)).or_insert([0; 4]);
        s[0] += 1; s[1] += e.blocks as u64; s[2] += hbad; s[3] += gbad;
        if hbad > 0 || gbad > 0 {
            println!("ORACLE BAD {} K{} {}x{}: host {hbad} values differ, GPU {gbad} values differ", e.module, e.bits, e.k, e.n);
            fails.push(format!("{} decode", e.module));
        }
        // linear: the Hadamard / suh / svh convention against exllamav3's get_weight_tensor order
        if let Some(lt) = &e.lin {
            let x = read_f32(&oracle.join(&lt.x))?;
            let y = read_f64(&oracle.join(&lt.y))?;
            ensure!(x.len() == lt.rows * e.k && y.len() == lt.rows * e.n, "{}: lin vector sizes", e.module);
            let r16 = lt.rows.min(16);
            let yd = run_dense(&cx, &dw, &x, r16, e.k, e.n, None)?;
            let e_dec = rel(&yd, &y[..r16 * e.n]);
            let yt = run_dense(&cx, &dw, &x, lt.rows, e.k, e.n, Some(WideSrc::Auto))?;
            let e_wt = rel(&yt, &y);
            let yr = run_dense(&cx, &dw, &x, lt.rows, e.k, e.n, Some(WideSrc::Recon))?;
            let e_wr = rel(&yr, &y);
            let worst = e_dec.max(e_wt).max(e_wr);
            // the activation is rounded to fp16 after H128(x . suh) (by design: exllamav3 does the same); a convention
            // error (wrong order, wrong side, missing 1/sqrt(128)) is O(1)
            let ok = worst < 2e-3;
            lin_lines.push(format!("  {} {} {} K{} {}x{}: rel-L2 vs x.W_full(fp64): decode class (16 rows) {e_dec:.2e}, wide trellis (40) {e_wt:.2e}, wide reconstruct (40) {e_wr:.2e}; exllamav3 fused reconstruct vs the formula {:.2} fp16 ulps",
                if ok { "ok  " } else { "FAIL" }, e.cls, e.module, e.bits, e.k, e.n, lt.fused_vs_formula_max_fp16_ulps));
            if !ok { fails.push(format!("{} linear", e.module)); }
        }
        // collect routed-expert triples (5120 -> inter -> 5120) for the MoE kernels
        if let Some(pos) = e.module.rfind(".w") {
            let base = &e.module[..pos];
            if base.contains(".ffn.experts.") {
                let wi = match &e.module[pos + 1..] { "w1" => 0, "w2" => 1, "w3" => 2, _ => 3 };
                if wi < 3 {
                    let wf: Vec<f32> = refw.iter().map(|b| h2f(*b)).collect();
                    let mut l2 = lin;
                    l2.w = wf.clone();
                    trip.entry(base.to_string()).or_insert([None, None, None])[wi] = Some((l2, wf));
                }
            }
        }
        if i % 25 == 0 { println!("  {}/{} {} K{} ({:.0} s)", i + 1, m.entries.len(), e.module, e.bits, t0.elapsed().as_secs_f64()); }
    }
    println!("== decode, bit for bit against exllamav3 ext.reconstruct ==");
    let (mut tb, mut th, mut tg) = (0u64, 0u64, 0u64);
    for ((c, k), s) in &per {
        println!("  {} {c} K{k}: {} tensors, {} blocks ({} weights); host decode {} differ; GPU reconstruct kernel {} differ",
                 if s[2] == 0 && s[3] == 0 { "ok  " } else { "FAIL" }, s[0], s[1], s[1] * 256, s[2], s[3]);
        tb += s[1]; th += s[2]; tg += s[3];
    }
    println!("== dense linears (Hadamard / suh / svh convention) ==");
    for l in &lin_lines { println!("{l}"); }
    // MoE kernels on single experts (decode path 4 rows, prefill path 40 rows, generic 4 rows)
    println!("== routed / DSpark experts through the MoE kernels, against the f64 chain on exllamav3's weights ==");
    let mut rng = Rng(0x51_0B5E);
    let mut nexp = 0;
    for (base, t) in &trip {
        let [Some(a), Some(b), Some(c)] = t else { continue };
        let inter = a.0.out;
        ensure!(a.0.inn == 5120 && b.0.out == 5120 && b.0.inn == inter && c.0.out == inter, "{base}: unexpected expert shapes");
        let ex = Ex { l: [a.0.clone_lin(&cx)?, b.0.clone_lin(&cx)?, c.0.clone_lin(&cx)?] };
        let rows_w = 40usize;
        let x: Vec<f32> = (0..rows_w * 5120).map(|_| rng.f()).collect();
        let mut refs: Vec<f64> = Vec::with_capacity(rows_w * 5120);
        for r in 0..rows_w { refs.extend(ref_expert(&x[r * 5120..(r + 1) * 5120], &ex, 0.75, 10.0)); }
        let yd = one_expert_moe(&cx, &ex, &x, 4, MoePath::Auto, inter)?;
        let yg = one_expert_moe(&cx, &ex, &x, 4, MoePath::Generic, inter)?;
        let yw = one_expert_moe(&cx, &ex, &x, rows_w, MoePath::Auto, inter)?;
        let (ed, eg, ew) = (rel(&yd, &refs[..4 * 5120]), rel(&yg, &refs[..4 * 5120]), rel(&yw, &refs));
        let same = yd.iter().zip(&yg).all(|(p, q)| p.to_bits() == q.to_bits()) && yd.iter().zip(&yw[..4 * 5120]).all(|(p, q)| p.to_bits() == q.to_bits());
        let ok = ed.max(eg).max(ew) < 1e-3 && same;
        println!("  {} {base} (inter {inter}, K{}): rel-L2 decode {ed:.2e} / generic {eg:.2e} / prefill {ew:.2e}; decode == generic == prefill rows bit for bit: {same}",
                 if ok { "ok  " } else { "FAIL" }, a.0.k2 / 2);
        if !ok { fails.push(format!("{base} moe")); }
        nexp += 1;
    }
    let _ = f2bf;
    println!("Q1 oracle: {} tensors, {tb} blocks ({} weights) decoded: host {th} differ, GPU {tg} differ; {} linear vectors; {nexp} expert triples; {:.0} s",
             m.entries.len(), tb * 256, lin_lines.len(), t0.elapsed().as_secs_f64());
    if !fails.is_empty() { bail!("{} failures: {}", fails.len(), fails.join("; ")); }
    if tb == 0 { bail!("no tensors checked"); }
    println!("DSV41_Q1_ORACLE_OK");
    Ok(())
}

impl Lin {
    /// A device copy of this linear with the given host weights (the oracle's exllamav3 decode).
    fn clone_lin(&self, cx: &Cx) -> Result<Lin> {
        let mut l = Lin::from_host(cx, self.k2, self.inn, self.out, self.tr.clone(), self.suh.clone(), self.svh.clone(), false)?;
        l.w = self.w.clone();
        Ok(l)
    }
}
