//! LOADFAST (owner order 2026-10-26): staged host images for the Naive/EXL3 loader.
//!
//! Before: `load_slice_on` read + le_u16-converted + sliced EVERY tensor of EVERY layer serially
//! on every boot (a 137 GB pack = ~16.5 s/layer/rank; on the head, rank 0's share came straight
//! off the slow model disk), and the result of all that CPU work was thrown away at shutdown.
//! Four changes, one seam:
//!
//! 1. **The staged-layer cache** (under the tp-cache root, beside the blobs): the post-slice,
//!    post-concat host image of a rank's layer set — written once per (pack, world, rank,
//!    n_layers), sha256-verified, atomically published per layer (an interrupted build resumes).
//!    A warm boot reads the image from local disk and pays H2D only.
//! 3. **Rank-0 staging**: after the first boot the cache IS the head's local stage — the head
//!    never touches the model HDD again (the cold pass reads it exactly once).
//! 2. **Parallel staging**: `workers` threads stage layers (read + convert + slice + concat)
//!    while the calling thread uploads in layer order.
//! 4. **Overlap**: staging of layer l+k runs under the upload of layer l; the pipeline is the
//!    prefetch.
//!
//! `stage_layer` is a faithful port of the old `load_layer`'s host side (every `ensure!` kept);
//! `GpuModel::upload_layer` (model.rs) is the device side. Nothing about serving changes: the
//! same GpuLayer is built from the same bytes — the cache round-trips the exact u16 buffers the
//! direct path produces (the unit tests assert byte equality).
//!
//! Failure posture (never refuse to boot): a bad/missing cache entry falls back to staging that
//! layer from the pack; a cache WRITE failure disables cache writes for the rest of the boot with
//! one clear line and continues. Flags: `--naive-load-cache on|off|auto` (auto == on; default on),
//! `--naive-load-verify on|off` (sha256 warm reads; default on), `--naive-load-workers N`
//! (default 4, 1 = the old serial path).

use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::config::AttnKind;
use super::model::{etp_of, f16_bits_of, f32_of};
use super::pack::NaivePack;
use super::shard::Shard;

// -------------------------------------------------------------------------------------------------
// Options
// -------------------------------------------------------------------------------------------------

/// Loader behaviour from the CLI (`--naive-load-*`). Defaults: cache on, verify on, 4 workers.
#[derive(Clone, Copy, Debug)]
pub struct LoadOpts {
    pub cache: bool,
    pub verify: bool,
    pub workers: usize,
}

impl Default for LoadOpts {
    fn default() -> Self { LoadOpts { cache: true, verify: true, workers: 4 } }
}

impl LoadOpts {
    pub fn from_opts() -> Self {
        let on_unless_off = |flag: bool, v: Option<String>| v.map(|s| !s.eq_ignore_ascii_case("off")).unwrap_or(flag);
        let workers = crate::opts::var(crate::opt!("naive-load-workers")).ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.clamp(1, 16))
            .unwrap_or(4);
        LoadOpts {
            cache: on_unless_off(true, crate::opts::var(crate::opt!("naive-load-cache")).ok()),
            verify: on_unless_off(true, crate::opts::var(crate::opt!("naive-load-verify")).ok()),
            workers,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Staged artifacts (the host side of the old load_layer)
// -------------------------------------------------------------------------------------------------

fn le_u16(raw: &[u8]) -> Vec<u16> { raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

/// One quad's post-slice host image: exactly the three buffers the old upload path htod'd.
pub struct StagedQuad {
    pub name: String, // display name, identical to the old GpuQuad names (slices annotated)
    pub k: usize,
    pub n: usize,
    pub bpb: usize,
    pub col_off: usize,
    pub tr: Vec<u16>,
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
}

impl StagedQuad {
    pub fn upload(&self, dev: &Arc<cudarc::driver::CudaDevice>) -> Result<super::gpu::GpuQuad> {
        let (ks, a1) = super::gpu::chain_plan(self.k, self.n);
        Ok(super::gpu::GpuQuad {
            name: self.name.clone(),
            tr: dev.htod_sync_copy(&self.tr)?,
            suh: dev.htod_sync_copy(&self.suh)?,
            svh: dev.htod_sync_copy(&self.svh)?,
            k: self.k, n: self.n, bpb: self.bpb, ks, a1,
        })
    }
}

/// The concatenated MoE expert planes of one layer (etp-sliced or whole experts, as this rank holds them).
pub struct StagedMoe {
    pub bpb: usize,
    pub g_tr: Vec<u16>, pub u_tr: Vec<u16>, pub d_tr: Vec<u16>,
    pub suh_g: Vec<u16>, pub svh_g: Vec<u16>,
    pub suh_u: Vec<u16>, pub svh_u: Vec<u16>,
    pub suh_d: Vec<u16>, pub svh_d: Vec<u16>,
}

/// One layer's complete host image (the pre-loop section too: embed/norm as plains, lm_head as
/// its single quad). Quads are consumed POSITIONALLY by the upload side via name-prefix lookup
/// (staging order is deterministic); plains are name-keyed.
pub struct StagedLayer {
    pub quads: Vec<StagedQuad>,
    pub plains: Vec<(String, Vec<u16>)>,   // f16-bit plains (norms, indexer k_norm, router, sinks)
    pub plains32: Vec<(String, Vec<f32>)>, // f32 plains (router bias)
    pub moe: Option<StagedMoe>,
}

impl StagedLayer {
    pub fn quad(&self, name_prefix: &str) -> Result<&StagedQuad> {
        self.quads.iter().find(|q| q.name.starts_with(name_prefix))
            .with_context(|| format!("staged image misses quad {name_prefix}"))
    }
    pub fn plain(&self, name: &str) -> Result<&[u16]> {
        self.plains.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
            .with_context(|| format!("staged image misses plain {name}"))
    }
    pub fn plain32(&self, name: &str) -> Result<&[f32]> {
        self.plains32.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
            .with_context(|| format!("staged image misses f32 plain {name}"))
    }
}

// --- staging: the faithful port of the old host-side work ----------------------------------------

fn stage_whole(pack: &NaivePack, q: &super::pack::Quad) -> Result<StagedQuad> {
    let tr = le_u16(&q.trellis.read_bytes(&pack.dir)?);
    let suh = le_u16(&q.suh.read_bytes(&pack.dir)?);
    let svh = le_u16(&q.svh.read_bytes(&pack.dir)?);
    ensure!(tr.len() == (q.k / 16) * (q.n / 16) * q.bits.words(), "{}: trellis length", q.name);
    Ok(StagedQuad { name: q.name.clone(), k: q.k, n: q.n, bpb: q.bits.words(), col_off: 0, tr, suh, svh })
}

/// A column (n) slice [a, b) rounded OUT to 128-blocks (the old `upload_quad_cols` host side).
fn stage_cols(pack: &NaivePack, q: &super::pack::Quad, a: usize, b: usize) -> Result<StagedQuad> {
    ensure!(a < b && b <= q.n && a % 16 == 0 && b % 16 == 0, "{}: column slice [{a}, {b}) of n {}", q.name, q.n);
    let a128 = a / 128 * 128;
    let b128 = ((b + 127) / 128 * 128).min(q.n);
    ensure!(b128 % 128 == 0 || b128 == q.n, "{}: n {} is not a 128-multiple", q.name, q.n);
    let words = q.bits.words();
    let (kb, nb) = (q.k / 16, q.n / 16);
    let tr = le_u16(&q.trellis.read_bytes(&pack.dir)?);
    ensure!(tr.len() == kb * nb * words, "{}: trellis length", q.name);
    let (ta, tb) = (a128 / 16, b128 / 16);
    let mut out = Vec::with_capacity(kb * (tb - ta) * words);
    for r in 0..kb {
        let row = &tr[r * nb * words..(r + 1) * nb * words];
        out.extend_from_slice(&row[ta * words..tb * words]);
    }
    let suh = le_u16(&q.suh.read_bytes(&pack.dir)?);
    let svh = le_u16(&q.svh.read_bytes(&pack.dir)?);
    ensure!(suh.len() == q.k && svh.len() == q.n, "{}: suh/svh length", q.name);
    Ok(StagedQuad {
        name: format!("{}[:, {a128}..{b128}]", q.name),
        k: q.k, n: b128 - a128, bpb: words, col_off: a - a128,
        tr: out, suh, svh: svh[a128..b128].to_vec(),
    })
}

/// A row (k) slice [a, b) (the old `upload_quad_rows` host side); a and b must be 128-multiples.
fn stage_rows(pack: &NaivePack, q: &super::pack::Quad, a: usize, b: usize) -> Result<StagedQuad> {
    ensure!(a < b && b <= q.k && a % 128 == 0 && b % 128 == 0, "{}: row slice [{a}, {b}) of k {} must be 128-aligned", q.name, q.k);
    let words = q.bits.words();
    let (kb, nb) = (q.k / 16, q.n / 16);
    let tr = le_u16(&q.trellis.read_bytes(&pack.dir)?);
    ensure!(tr.len() == kb * nb * words, "{}: trellis length", q.name);
    let (ra, rb) = (a / 16, b / 16);
    let suh = le_u16(&q.suh.read_bytes(&pack.dir)?);
    let svh = le_u16(&q.svh.read_bytes(&pack.dir)?);
    ensure!(suh.len() == q.k && svh.len() == q.n, "{}: suh/svh length", q.name);
    Ok(StagedQuad {
        name: format!("{}[{a}..{b}, :]", q.name),
        k: b - a, n: q.n, bpb: words, col_off: 0,
        tr: tr[ra * nb * words..rb * nb * words].to_vec(),
        suh: suh[a..b].to_vec(), svh,
    })
}

/// The MoE interm_div guard. Reads only the staged planes, so the upload side runs it on BOTH
/// paths — a cold load validates the pack, a warm load validates the cache.
pub fn check_interm_div(l: usize, s: &StagedMoe, div: f32) -> Result<()> {
    let mean_abs = |v: &[u16]| v.iter().map(|b| half::f16::from_bits(*b).to_f32().abs() as f64).sum::<f64>() / v.len().max(1) as f64;
    let ratio = mean_abs(&s.suh_u) / mean_abs(&s.suh_g).max(1e-12);
    if div != 1.0 {
        ensure!((ratio * div as f64 - 1.0).abs() < 0.5,
                "layer {l}: config says the pack's up_proj is stored /{div} (exllamav3 interm_div) but its input scale is {ratio:.4} of gate_proj's (expected ~{:.4})", 1.0 / div as f64);
        crate::dprintln!("  layer {l}: routed up_proj stored /{div} (up/gate input-scale ratio {ratio:.4}); routing weights scaled x{div} (exllamav3 naive_n05 interm_div)");
    } else {
        ensure!(ratio > 0.1, "layer {l}: the pack's up_proj input scale is {ratio:.4} of gate_proj's: the converter divided it (interm_div) but the config has no divisor for this layer");
    }
    Ok(())
}

/// Stage one layer: every read, le_u16, slice and concat the old `load_layer` did host-side, in
/// the same order with the same `ensure!`s — no device work.
pub fn stage_layer(pack: &NaivePack, l: usize, shard: &Shard) -> Result<StagedLayer> {
    let c = &pack.cfg;
    let h = c.hidden_size;
    let hd = c.head_dim;
    let p = format!("model.layers.{l}.");
    let kv_total = c.kv_heads(l);
    let (q0, nh_l) = shard.q_heads(c.num_heads)?;
    let (k0, kv_l) = shard.kv_heads(c.num_heads, kv_total)?;
    let mut quads: Vec<StagedQuad> = Vec::new();
    let mut plains: Vec<(String, Vec<u16>)> = Vec::new();
    let mut plains32: Vec<(String, Vec<f32>)> = Vec::new();
    let qd = |name: &str| pack.quad(&format!("{p}{name}"));

    // attention projections: q columns of this rank's heads, k/v columns of its kv heads, o rows
    if shard.world == 1 {
        for w in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
            quads.push(stage_whole(pack, qd(w)?)?);
        }
    } else {
        quads.push(stage_cols(pack, qd("self_attn.q_proj")?, q0 * hd, (q0 + nh_l) * hd)?);
        quads.push(stage_cols(pack, qd("self_attn.k_proj")?, k0 * hd, (k0 + kv_l) * hd)?);
        quads.push(stage_cols(pack, qd("self_attn.v_proj")?, k0 * hd, (k0 + kv_l) * hd)?);
        let oq = qd("self_attn.o_proj")?;
        // a shipped shard carries o_proj PRE-SLICED to this rank's rows (k == nh_l * v_head_dim)
        quads.push(if oq.k == nh_l * c.v_head_dim { stage_whole(pack, oq)? } else { stage_rows(pack, oq, q0 * c.v_head_dim, (q0 + nh_l) * c.v_head_dim)? });
    }
    // DSA indexer (replicated quads + two f16 plains)
    if c.attn[l] == AttnKind::Dsa {
        for w in ["self_attn.indexer.wq", "self_attn.indexer.wk", "self_attn.indexer.weights_proj"] {
            quads.push(stage_whole(pack, qd(w)?)?);
        }
        plains.push((format!("{p}self_attn.indexer.k_norm.weight"), f16_bits_of(pack, &format!("{p}self_attn.indexer.k_norm.weight"))?));
        plains.push((format!("{p}self_attn.indexer.k_norm.bias"), f16_bits_of(pack, &format!("{p}self_attn.indexer.k_norm.bias"))?));
    }
    // attention sink (this rank's heads' slice)
    if c.has_sink(l) {
        let all = f16_bits_of(pack, &format!("{p}self_attn.attention_sink_bias"))?;
        plains.push((format!("{p}self_attn.attention_sink_bias"), all[q0..q0 + nh_l].to_vec()));
    }
    // MLP
    let moe = if c.moe[l] {
        let (e, mi) = (c.num_experts, c.moe_intermediate_size);
        let etp = etp_of(pack, shard);
        let mi_l = if etp { mi / shard.world } else { mi };
        ensure!(mi_l % 128 == 0, "expert TP: the intermediate slice {mi_l} is not 128-aligned at world {}", shard.world);
        let (ia, ib) = (shard.rank * mi_l, (shard.rank + 1) * mi_l);
        let experts: Vec<usize> = if etp { (0..e).collect() } else { shard.local_experts(e) };
        let mut s = StagedMoe {
            bpb: 0,
            g_tr: Vec::new(), u_tr: Vec::new(), d_tr: Vec::new(),
            suh_g: Vec::new(), svh_g: Vec::new(), suh_u: Vec::new(), svh_u: Vec::new(), suh_d: Vec::new(), svh_d: Vec::new(),
        };
        for &ex in &experts {
            let q = |w: &str| pack.quad(&format!("{p}mlp.experts.{ex}.{w}"));
            let (g, u, d) = (q("gate_proj")?, q("up_proj")?, q("down_proj")?);
            // whole experts (the full pack, or the expert-parallel deal) or pre-sliced ones (an expert-TP shard)
            let pre = etp && g.n == mi_l && mi_l != mi;
            let (gn, dk) = if pre { (mi_l, mi_l) } else { (mi, mi) };
            ensure!(g.k == h && g.n == gn && u.k == h && u.n == gn && d.k == dk && d.n == h,
                    "expert geometry @{l} e{ex}: gate {}x{} up {}x{} down {}x{}", g.k, g.n, u.k, u.n, d.k, d.n);
            ensure!(g.bits == u.bits && g.bits == d.bits && (s.bpb == 0 || s.bpb == g.bits.words()), "expert widths @{l} e{ex}");
            s.bpb = g.bits.words();
            let (gt, ut, dt) = (le_u16(&g.trellis.read_bytes(&pack.dir)?), le_u16(&u.trellis.read_bytes(&pack.dir)?), le_u16(&d.trellis.read_bytes(&pack.dir)?));
            let (gsv, usv, dsu) = (le_u16(&g.svh.read_bytes(&pack.dir)?), le_u16(&u.svh.read_bytes(&pack.dir)?), le_u16(&d.suh.read_bytes(&pack.dir)?));
            if etp && !pre {
                // slice here (the head's whole pack): gate/up keep tile columns [ia/16, ib/16) of
                // every tile row and that svh range; down keeps tile rows [ia/16, ib/16) and that suh range
                let (kb, nb, w) = (h / 16, mi / 16, s.bpb);
                let cols = |t: &[u16]| -> Vec<u16> {
                    let mut o = Vec::with_capacity(kb * (mi_l / 16) * w);
                    for r in 0..kb { o.extend_from_slice(&t[(r * nb + ia / 16) * w..(r * nb + ib / 16) * w]); }
                    o
                };
                s.g_tr.extend(cols(&gt)); s.u_tr.extend(cols(&ut));
                s.d_tr.extend_from_slice(&dt[(ia / 16) * (h / 16) * w..(ib / 16) * (h / 16) * w]);
                s.svh_g.extend_from_slice(&gsv[ia..ib]); s.svh_u.extend_from_slice(&usv[ia..ib]);
                s.suh_d.extend_from_slice(&dsu[ia..ib]);
            } else {
                s.g_tr.extend(gt); s.u_tr.extend(ut); s.d_tr.extend(dt);
                s.svh_g.extend(gsv); s.svh_u.extend(usv); s.suh_d.extend(dsu);
            }
            s.suh_g.extend(le_u16(&g.suh.read_bytes(&pack.dir)?));
            s.suh_u.extend(le_u16(&u.suh.read_bytes(&pack.dir)?));
            s.svh_d.extend(le_u16(&d.svh.read_bytes(&pack.dir)?));
        }
        plains.push((format!("{p}mlp.gate.weight"), f16_bits_of(pack, &format!("{p}mlp.gate.weight"))?));
        plains32.push((format!("{p}mlp.gate.e_score_correction_bias"), f32_of(pack, &format!("{p}mlp.gate.e_score_correction_bias"))?));
        Some(s)
    } else if shard.world == 1 {
        for w in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] { quads.push(stage_whole(pack, qd(w)?)?); }
        None
    } else {
        let (i0, il) = shard.span(c.intermediate_size)?;
        ensure!(i0 % 128 == 0 && il % 128 == 0, "dense intermediate slice not 128-aligned");
        quads.push(stage_cols(pack, qd("mlp.gate_proj")?, i0, i0 + il)?);
        quads.push(stage_cols(pack, qd("mlp.up_proj")?, i0, i0 + il)?);
        let dq = qd("mlp.down_proj")?;
        quads.push(if dq.k == il { stage_whole(pack, dq)? } else { stage_rows(pack, dq, i0, i0 + il)? });
        None
    };
    // the two norms every layer carries
    plains.push((format!("{p}input_layernorm.weight"), f16_bits_of(pack, &format!("{p}input_layernorm.weight"))?));
    plains.push((format!("{p}post_attention_layernorm.weight"), f16_bits_of(pack, &format!("{p}post_attention_layernorm.weight"))?));
    Ok(StagedLayer { quads, plains, plains32, moe })
}

/// Stage the pre-loop section: embed (bf16 bits), final norm (f16 bits), this rank's lm_head quad.
pub fn stage_pre(pack: &NaivePack, shard: &Shard) -> Result<StagedLayer> {
    let m = pack.plain_tensor("model.embed_tokens.weight")?;
    ensure!(m.dtype == "BF16", "embed dtype {}", m.dtype);
    let embed = le_u16(&m.read_bytes(&pack.dir)?);
    let norm = f16_bits_of(pack, "model.norm.weight")?;
    let q = pack.quad("lm_head")?;
    let lm_head = if shard.world == 1 {
        stage_whole(pack, q)?
    } else {
        let (a, n) = shard.span(pack.cfg.vocab_size)?;
        ensure!(a % 128 == 0 && n % 128 == 0, "vocabulary slice not 128-aligned");
        stage_cols(pack, q, a, a + n)?
    };
    Ok(StagedLayer {
        quads: vec![lm_head],
        plains: vec![("model.embed_tokens.weight".into(), embed), ("model.norm.weight".into(), norm)],
        plains32: Vec::new(),
        moe: None,
    })
}

// -------------------------------------------------------------------------------------------------
// The staged-layer cache
// -------------------------------------------------------------------------------------------------

const CACHE_VERSION: u32 = 1;

/// Same root as the B32 blob cache (mirrors cluster.rs `cache_root`; kept in sync deliberately).
fn cache_root() -> PathBuf {
    if let Ok(p) = crate::opts::var(crate::opt!("tp-cache")) { return PathBuf::from(p); }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache/gb10_tp")
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

/// Identity of everything the staged image depends on. Any change invalidates the cache.
pub fn fingerprint(pack: &NaivePack, n_layers: usize, shard: &Shard) -> BTreeMap<String, String> {
    let c = &pack.cfg;
    let mut f = BTreeMap::new();
    f.insert("version".into(), CACHE_VERSION.to_string());
    f.insert("pack_dir".into(), Path::new(&pack.dir).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    f.insert("total_bytes".into(), pack.total_bytes.to_string());
    f.insert("total_tensors".into(), pack.total_tensors.to_string());
    f.insert("world".into(), shard.world.to_string());
    f.insert("rank".into(), shard.rank.to_string());
    f.insert("n_layers".into(), n_layers.to_string());
    f.insert("quant".into(), format!("{:?}", pack.quant));
    f.insert("cfg_layers".into(), c.num_layers.to_string());
    f.insert("hidden".into(), c.hidden_size.to_string());
    f.insert("vocab".into(), c.vocab_size.to_string());
    f.insert("experts".into(), c.num_experts.to_string());
    f.insert("moe_inter".into(), c.moe_intermediate_size.to_string());
    f.insert("etp".into(), etp_of(pack, shard).to_string());
    f.insert("attn".into(), c.attn.iter().map(|a| if *a == AttnKind::Dsa { "d" } else { "s" }).collect::<String>());
    f.insert("moe".into(), c.moe.iter().map(|m| if *m { "1" } else { "0" }).collect::<String>());
    f
}

#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
struct FileEntry { sha256: String, bytes: u64 }

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct Manifest {
    version: u32,
    fingerprint: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pre: Option<FileEntry>,
    #[serde(default)]
    layers: BTreeMap<usize, FileEntry>,
}

// --- binary image format -------------------------------------------------------------------------
// magic u32 'NLQ1' | n_quads u32 | per quad { name(u32 len + bytes), k, n, bpb, col_off u32,
// tr/suh/svh len u64 } | n_f16 u32 | per { name, len u64 } | n_f32 u32 | per { name, len u64 } |
// moe u8 | if 1 { bpb u32, 9x len u64 } | payloads (u16 LE / f32 LE, quads then plains then moe)

struct Writer<W: Write> { w: W, h: Sha256, bytes: u64 }
impl<W: Write> Writer<W> {
    fn new(w: W) -> Self { Writer { w, h: Sha256::new(), bytes: 0 } }
    fn put(&mut self, b: &[u8]) -> std::io::Result<()> { self.h.update(b); self.bytes += b.len() as u64; self.w.write_all(b) }
    fn u32(&mut self, v: u32) -> std::io::Result<()> { self.put(&v.to_le_bytes()) }
    fn u64(&mut self, v: u64) -> std::io::Result<()> { self.put(&v.to_le_bytes()) }
    fn u8b(&mut self, v: u8) -> std::io::Result<()> { self.put(&[v]) }
    fn name(&mut self, v: &str) -> std::io::Result<()> { self.u32(v.len() as u32)?; self.put(v.as_bytes()) }
    fn u16s(&mut self, v: &[u16]) -> std::io::Result<()> { for x in v { self.put(&x.to_le_bytes())?; } Ok(()) }
    fn f32s(&mut self, v: &[f32]) -> std::io::Result<()> { for x in v { self.put(&x.to_bits().to_le_bytes())?; } Ok(()) }
    /// Flush and return (sha256 hex, byte count) of everything written.
    fn finish(mut self) -> std::io::Result<(String, u64)> {
        self.w.flush()?;
        Ok((hex(&self.h.finalize()), self.bytes))
    }
}

struct Reader<R: Read> { r: R, h: Sha256, bytes: u64 }
impl<R: Read> Reader<R> {
    fn new(r: R) -> Self { Reader { r, h: Sha256::new(), bytes: 0 } }
    fn take(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.r.read_exact(&mut *buf)?;
        self.h.update(&*buf); self.bytes += buf.len() as u64; Ok(())
    }
    fn u32(&mut self) -> std::io::Result<u32> { let mut b = [0u8; 4]; self.take(&mut b)?; Ok(u32::from_le_bytes(b)) }
    fn u64(&mut self) -> std::io::Result<u64> { let mut b = [0u8; 8]; self.take(&mut b)?; Ok(u64::from_le_bytes(b)) }
    fn u8b(&mut self) -> std::io::Result<u8> { let mut b = [0u8; 1]; self.take(&mut b)?; Ok(b[0]) }
    fn name(&mut self) -> std::io::Result<String> {
        let n = self.u32()? as usize;
        if n >= 4096 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("cache: absurd name length {n}")));
        }
        let mut b = vec![0u8; n]; self.take(&mut b)?;
        String::from_utf8(b).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    fn u16s(&mut self, len: usize) -> std::io::Result<Vec<u16>> {
        let mut raw = vec![0u8; len * 2]; self.take(&mut raw)?;
        Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn f32s(&mut self, len: usize) -> std::io::Result<Vec<f32>> {
        let mut raw = vec![0u8; len * 4]; self.take(&mut raw)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect())
    }
    fn digest(&mut self) -> String { self.h.clone().finalize().iter().map(|x| format!("{x:02x}")).collect() }
}

fn encode_staged(s: &StagedLayer, w: Writer<impl Write>) -> Result<(String, u64)> {
    let mut w = w;
    w.u32(0x4E4C5131)?; // 'NLQ1'
    w.u32(s.quads.len() as u32)?;
    for q in &s.quads {
        w.name(&q.name)?;
        w.u32(q.k as u32)?; w.u32(q.n as u32)?; w.u32(q.bpb as u32)?; w.u32(q.col_off as u32)?;
        w.u64(q.tr.len() as u64)?; w.u64(q.suh.len() as u64)?; w.u64(q.svh.len() as u64)?;
    }
    w.u32(s.plains.len() as u32)?;
    for (n, v) in &s.plains { w.name(n)?; w.u64(v.len() as u64)?; }
    w.u32(s.plains32.len() as u32)?;
    for (n, v) in &s.plains32 { w.name(n)?; w.u64(v.len() as u64)?; }
    w.u8b(s.moe.is_some() as u8)?;
    if let Some(m) = &s.moe {
        w.u32(m.bpb as u32)?;
        for v in [&m.g_tr, &m.u_tr, &m.d_tr, &m.suh_g, &m.svh_g, &m.suh_u, &m.svh_u, &m.suh_d, &m.svh_d] { w.u64(v.len() as u64)?; }
    }
    for q in &s.quads { w.u16s(&q.tr)?; w.u16s(&q.suh)?; w.u16s(&q.svh)?; }
    for (_, v) in &s.plains { w.u16s(v)?; }
    for (_, v) in &s.plains32 { w.f32s(v)?; }
    if let Some(m) = &s.moe {
        for v in [&m.g_tr, &m.u_tr, &m.d_tr, &m.suh_g, &m.svh_g, &m.suh_u, &m.svh_u, &m.suh_d, &m.svh_d] { w.u16s(v)?; }
    }
    Ok(w.finish()?)
}

fn decode_staged(r: Reader<impl Read>, expect: Option<&FileEntry>, verify: bool) -> Result<StagedLayer> {
    let mut r = r;
    let magic = r.u32()?;
    ensure!(magic == 0x4E4C5131, "cache: bad magic {magic:#x}");
    let n_quads = r.u32()? as usize;
    ensure!(n_quads < 4096, "cache: absurd quad count");
    let mut heads = Vec::with_capacity(n_quads);
    for _ in 0..n_quads {
        let name = r.name()?;
        let k = r.u32()? as usize; let n = r.u32()? as usize; let bpb = r.u32()? as usize; let col_off = r.u32()? as usize;
        let tr = r.u64()? as usize; let suh = r.u64()? as usize; let svh = r.u64()? as usize;
        heads.push((name, k, n, bpb, col_off, tr, suh, svh));
    }
    let n16 = r.u32()? as usize;
    ensure!(n16 < 4096, "cache: absurd plain count");
    let mut p16 = Vec::with_capacity(n16);
    for _ in 0..n16 { let name = r.name()?; let len = r.u64()? as usize; p16.push((name, len)); }
    let n32 = r.u32()? as usize;
    let mut p32 = Vec::with_capacity(n32);
    for _ in 0..n32 { let name = r.name()?; let len = r.u64()? as usize; p32.push((name, len)); }
    let has_moe = r.u8b()? == 1;
    let moe_bpb = if has_moe { r.u32()? as usize } else { 0 };
    let mut mlens = [0usize; 9];
    if has_moe { for m in mlens.iter_mut() { *m = r.u64()? as usize; } }
    let mut quads = Vec::with_capacity(n_quads);
    for (name, k, n, bpb, col_off, tr, suh, svh) in heads {
        quads.push(StagedQuad { name, k, n, bpb, col_off, tr: r.u16s(tr)?, suh: r.u16s(suh)?, svh: r.u16s(svh)? });
    }
    let mut plains = Vec::with_capacity(n16);
    for (name, len) in p16 { plains.push((name, r.u16s(len)?)); }
    let mut plains32 = Vec::with_capacity(n32);
    for (name, len) in p32 { plains32.push((name, r.f32s(len)?)); }
    let moe = if has_moe {
        let mut v = Vec::with_capacity(9);
        for len in mlens { v.push(r.u16s(len)?); }
        let mut it = v.into_iter();
        Some(StagedMoe {
            bpb: moe_bpb,
            g_tr: it.next().unwrap(), u_tr: it.next().unwrap(), d_tr: it.next().unwrap(),
            suh_g: it.next().unwrap(), svh_g: it.next().unwrap(),
            suh_u: it.next().unwrap(), svh_u: it.next().unwrap(),
            suh_d: it.next().unwrap(), svh_d: it.next().unwrap(),
        })
    } else { None };
    if verify {
        if let Some(e) = expect {
            let got = r.digest();
            ensure!(got == e.sha256, "cache: sha256 mismatch (expected {}, got {got})", e.sha256);
            ensure!(r.bytes == e.bytes, "cache: size mismatch (expected {}, got {})", e.bytes, r.bytes);
        }
    }
    Ok(StagedLayer { quads, plains, plains32, moe })
}

/// Atomically publish one staged image as `path`; returns its manifest entry (sha256 + size).
fn write_image(path: &Path, s: &StagedLayer) -> Result<FileEntry> {
    if let Some(dir) = path.parent() { std::fs::create_dir_all(dir)?; }
    let tmp = path.with_extension("tmp");
    let f = std::fs::File::create(&tmp).with_context(|| format!("cache write {}", tmp.display()))?;
    let (sha256, bytes) = encode_staged(s, Writer::new(std::io::BufWriter::with_capacity(1 << 20, f)))
        .with_context(|| format!("cache encode {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("cache publish {}", path.display()))?;
    Ok(FileEntry { sha256, bytes })
}

fn read_image(path: &Path, expect: Option<&FileEntry>, verify: bool) -> Result<StagedLayer> {
    let f = std::fs::File::open(path).with_context(|| format!("cache read {}", path.display()))?;
    decode_staged(Reader::new(std::io::BufReader::with_capacity(1 << 20, f)), expect, verify)
}

// -------------------------------------------------------------------------------------------------
// The pipeline: workers stage out of order, the consumer uploads in order (scoped threads)
// -------------------------------------------------------------------------------------------------

/// Out-of-order hand-off: producers submit completed items by index, `next()` returns them
/// strictly in order. Submitting NEVER blocks — the memory bound lives in `ClaimGate`, on the
/// CLAIM side, because a bound at submit time can block the very item the consumer is waiting
/// for (the LOADFAST unit-test deadlock: reverse-scheduled producers fill the bound with later
/// layers while the consumer waits for an earlier one).
struct OrderBuf<T> {
    n: usize,
    done: (Mutex<BTreeMap<usize, T>>, Condvar),
    err: (Mutex<Option<String>>, Condvar),
}

impl<T> OrderBuf<T> {
    fn new(n: usize) -> Self {
        OrderBuf {
            n,
            done: (Mutex::new(BTreeMap::new()), Condvar::new()),
            err: (Mutex::new(None), Condvar::new()),
        }
    }
    fn set_err(&self, msg: String) {
        *self.err.0.lock().unwrap() = Some(msg);
        self.err.1.notify_all();
        self.done.1.notify_all();
    }
    /// Producer side: submit item `i` (non-blocking).
    fn submit(&self, i: usize, v: T) {
        self.done.0.lock().unwrap().insert(i, v);
        self.done.1.notify_all();
    }
    /// Consumer side: items in order; `Ok(None)` = all consumed.
    fn next(&self, want: usize) -> Result<Option<T>> {
        let mut g = self.done.0.lock().unwrap();
        loop {
            if let Some(v) = g.remove(&want) {
                return Ok(Some(v));
            }
            if let Some(e) = self.err.0.lock().unwrap().clone() { bail!("staging failed: {e}"); }
            if want >= self.n && g.is_empty() { return Ok(None); }
            g = self.done.1.wait(g).unwrap();
        }
    }
    fn note_err(&self) -> bool { self.err.0.lock().unwrap().is_some() }
}

/// The staging look-ahead bound (a memory cap). Claims are handed out strictly in order
/// 0,1,2,…, so the layer the consumer is waiting for is always already claimed: the bound can
/// delay later layers but never the waited-on one — deadlock-free under any worker scheduling.
struct ClaimGate {
    st: Mutex<(usize /* next */, usize /* outstanding */)>,
    cv: Condvar,
    failed: AtomicBool,
}

impl ClaimGate {
    fn new() -> Self { ClaimGate { st: Mutex::new((0, 0)), cv: Condvar::new(), failed: AtomicBool::new(false) } }
    /// Next claimable layer, waiting while `bound` layers are outstanding; None = done or failed.
    fn claim(&self, n: usize, bound: usize) -> Option<usize> {
        let mut g = self.st.lock().unwrap();
        loop {
            if self.failed.load(Ordering::Relaxed) { return None; }
            if g.0 >= n { return None; }
            if g.1 >= bound { g = self.cv.wait(g).unwrap(); continue; }
            let l = g.0; g.0 += 1; g.1 += 1;
            return Some(l);
        }
    }
    /// The consumer finished one layer: release a slot.
    fn consumed(&self) {
        let mut g = self.st.lock().unwrap();
        g.1 = g.1.saturating_sub(1);
        drop(g);
        self.cv.notify_all();
    }
    fn fail(&self) { self.failed.store(true, Ordering::Relaxed); self.cv.notify_all(); }
}

struct Shared<'a> {
    pack: &'a NaivePack,
    shard: Shard,
    n: usize,
    opts: LoadOpts,
    gate: ClaimGate,
    order: OrderBuf<StagedLayer>,
    manifest: Mutex<Manifest>,
    dir: Option<PathBuf>,
    warm: AtomicUsize,
    cold: AtomicUsize,
    cache_writes: AtomicBool,
}

fn save_manifest(dir: &Path, m: &Manifest) {
    let p = dir.join("manifest.json");
    let tmp = dir.join("manifest.json.tmp");
    if let Ok(b) = serde_json::to_vec_pretty(m) {
        if std::fs::write(&tmp, &b).and_then(|_| std::fs::rename(&tmp, &p)).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

impl<'a> Shared<'a> {
    /// One layer: warm read when the manifest has a valid entry, else cold stage + write-through.
    /// Errors are recorded on the OrderBuf (fatal, like the old loader's).
    fn stage_one(&self, l: usize) {
        let cached = {
            let mut m = self.manifest.lock().unwrap();
            if let Some(d) = self.dir.as_ref() {
                let p = d.join(format!("layer-{l:05}.bin"));
                if let Some(e) = m.layers.get(&l).cloned() {
                    if p.exists() {
                        match read_image(&p, Some(&e), self.opts.verify) {
                            Ok(s) => Some(s),
                            Err(err) => {
                                crate::dprintln!("[naive-load] cache layer {l} unreadable ({err:#}); staging cold");
                                m.layers.remove(&l);
                                None
                            }
                        }
                    } else { m.layers.remove(&l); None }
                } else { None }
            } else { None }
        };
        let s = match cached {
            Some(s) => { self.warm.fetch_add(1, Ordering::Relaxed); return self.order.submit(l, s); }
            None => match stage_layer(self.pack, l, &self.shard) {
                Ok(s) => s,
                Err(e) => { self.order.set_err(format!("{e:#}")); self.gate.fail(); return; }
            },
        };
        self.cold.fetch_add(1, Ordering::Relaxed);
        if let Some(d) = self.dir.as_ref() {
            if self.cache_writes.load(Ordering::Relaxed) {
                let p = d.join(format!("layer-{l:05}.bin"));
                match write_image(&p, &s) {
                    Ok(e) => {
                        let mut m = self.manifest.lock().unwrap();
                        m.layers.insert(l, e);
                        save_manifest(d, &m);
                    }
                    Err(err) => {
                        self.cache_writes.store(false, Ordering::Relaxed);
                        println!("[naive-load] cache writes disabled for this boot ({:#})", err);
                    }
                }
            }
        }
        self.order.submit(l, s);
    }
}

/// What the consumer receives: the pre-loop image first, then each layer in order.
pub enum Item<'a> {
    Pre(&'a StagedLayer),
    Layer(usize, &'a StagedLayer),
}

pub struct Stats { pub warm: usize, pub cold: usize, pub secs: f32, pub cache: Option<PathBuf> }

/// Stage `n_layers` layers of `pack` for `shard` and hand them to `consume` in order — the
/// pre-loop image first (embed/norm/lm_head), then layers 0..n. Workers stage ahead (bounded),
/// the cache is read/written per the options, and the summary line is printed at the end.
pub fn run<F>(pack: &NaivePack, n_layers: usize, shard: Shard, opts: &LoadOpts, mut consume: F) -> Result<Stats>
where F: FnMut(Item<'_>) -> Result<()>
{
    let t0 = std::time::Instant::now();
    let fp = fingerprint(pack, n_layers, &shard);
    let key = format!("{}-w{}r{}L{}",
                      fp.get("pack_dir").map(|s| s.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.', "-")).unwrap_or_default(),
                      fp["world"], fp["rank"], fp["n_layers"]);
    let dir = if opts.cache { Some(cache_root().join("naive-load").join(key)) } else { None };
    let mut manifest = Manifest { version: CACHE_VERSION, fingerprint: fp.clone(), ..Default::default() };
    if let Some(d) = dir.as_ref() {
        match std::fs::read(d.join("manifest.json")).ok().and_then(|b| serde_json::from_slice::<Manifest>(&b).ok()) {
            Some(m) if m.version == CACHE_VERSION && m.fingerprint == fp => {
                println!("[naive-load] cache {}: manifest valid, {} of {n_layers} layers present", d.display(), m.layers.len().min(n_layers));
                manifest = m;
            }
            _ => {
                // stale or absent: start clean (a different fingerprint's files are dead weight)
                let _ = std::fs::remove_dir_all(d);
                let _ = std::fs::create_dir_all(d);
                println!("[naive-load] cache {}: cold, building (verify {}, {} workers)", d.display(), if opts.verify { "on" } else { "off" }, opts.workers);
            }
        }
    } else {
        println!("[naive-load] cache off (--naive-load-cache off)");
    }

    // the pre-loop image (warm when possible, else staged + written through) — single-threaded
    // phase, so the manifest is a plain local here
    let pre = {
        let mut warm_pre = None;
        if let (Some(d), Some(e)) = (dir.as_ref(), manifest.pre.clone()) {
            let p = d.join("pre.bin");
            if p.exists() {
                match read_image(&p, Some(&e), opts.verify) {
                    Ok(s) => warm_pre = Some(s),
                    Err(err) => {
                        crate::dprintln!("[naive-load] cache pre unreadable ({err:#}); staging cold");
                        manifest.pre = None;
                    }
                }
            } else { manifest.pre = None; }
        }
        match warm_pre {
            Some(s) => s,
            None => {
                let s = stage_pre(pack, &shard)?;
                if let Some(d) = dir.as_ref() {
                    match write_image(&d.join("pre.bin"), &s) {
                        Ok(e) => {
                            manifest.pre = Some(e);
                            save_manifest(d, &manifest);
                        }
                        Err(err) => println!("[naive-load] cache writes disabled for this boot ({:#})", err),
                    }
                }
                s
            }
        }
    };
    consume(Item::Pre(&pre))?;

    let shared = Shared {
        pack, shard, n: n_layers, opts: *opts,
        gate: ClaimGate::new(),
        order: OrderBuf::new(n_layers),
        manifest: Mutex::new(manifest),
        dir,
        warm: AtomicUsize::new(0),
        cold: AtomicUsize::new(0),
        cache_writes: AtomicBool::new(true),
    };
    let mut consume_err: Option<anyhow::Error> = None;
    // BASICS (owner 2026-11-02): layers handed up, as a percentage at the normal level
    let prog = crate::boot_report::Progress::new("loading weights", n_layers as u64, crate::boot_report::Unit::Items("layers"));
    if opts.workers <= 1 {
        // the old serial shape: stage, then hand up, one layer at a time
        for l in 0..n_layers {
            shared.stage_one(l);
            match shared.order.next(l) {
                Ok(Some(s)) => {
                    if let Err(e) = consume(Item::Layer(l, &s)) { consume_err = Some(e); break; }
                    prog.add(1);
                }
                Ok(None) => break,
                Err(e) => { consume_err = Some(e); break; }
            }
        }
    } else {
        std::thread::scope(|scope| {
            for _ in 0..opts.workers {
                scope.spawn(|| loop {
                    let l = match shared.gate.claim(shared.n, shared.opts.workers + 2) { Some(l) => l, None => break };
                    if shared.order.note_err() { break; }
                    shared.stage_one(l);
                });
            }
            for l in 0..n_layers {
                match shared.order.next(l) {
                    Ok(Some(s)) => {
                        if let Err(e) = consume(Item::Layer(l, &s)) {
                            consume_err = Some(e);
                            shared.order.set_err("upload failed".into());
                            shared.gate.fail();
                            break;
                        }
                        shared.gate.consumed();
                        prog.add(1);
                    }
                    Ok(None) => break,
                    Err(e) => { consume_err = Some(e); shared.gate.fail(); break; }
                }
            }
        });
    }
    if let Some(e) = consume_err { return Err(e); }
    prog.finish();
    let stats = Stats {
        warm: shared.warm.load(Ordering::Relaxed),
        cold: shared.cold.load(Ordering::Relaxed),
        secs: t0.elapsed().as_secs_f32(),
        cache: shared.dir.clone(),
    };
    match stats.cache.as_ref() {
        Some(d) => println!("[naive-load] {n_layers}/{n_layers} layers staged in {:.1} s (warm {} / built {}; cache {})",
                            stats.secs, stats.warm, stats.cold, d.display()),
        None => println!("[naive-load] {n_layers}/{n_layers} layers staged in {:.1} s (cache off)", stats.secs),
    }
    Ok(stats)
}

// -------------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> StagedLayer {
        let q = |name: &str, k: usize, n: usize| StagedQuad {
            name: name.into(), k, n, bpb: 56, col_off: 7,
            tr: (0..k * n / 16 * 56 / 16).map(|i| (i % 65536) as u16).collect(),
            suh: (0..k).map(|i| (i * 3) as u16).collect(),
            svh: (0..n).map(|i| (i * 5 + 1) as u16).collect(),
        };
        StagedLayer {
            quads: vec![q("model.layers.0.self_attn.q_proj[:, 0..128]", 128, 128), q("model.layers.0.self_attn.o_proj", 256, 128)],
            plains: vec![("model.layers.0.input_layernorm.weight".into(), (0..64u16).collect())],
            plains32: vec![("model.layers.0.mlp.gate.e_score_correction_bias".into(), vec![0.25f32, -1.5, 3.0])],
            moe: Some(StagedMoe {
                bpb: 56,
                g_tr: vec![1, 2, 3, 65535], u_tr: vec![4, 5], d_tr: vec![9, 8, 7],
                suh_g: vec![100, 200], svh_g: vec![300, 400],
                suh_u: vec![11, 12], svh_u: vec![13, 14],
                suh_d: vec![15, 16], svh_d: vec![17, 18],
            }),
        }
    }

    fn assert_same(a: &StagedLayer, b: &StagedLayer) {
        assert_eq!(a.quads.len(), b.quads.len());
        for (x, y) in a.quads.iter().zip(&b.quads) {
            assert_eq!((x.name.as_str(), x.k, x.n, x.bpb, x.col_off, &x.tr, &x.suh, &x.svh),
                       (y.name.as_str(), y.k, y.n, y.bpb, y.col_off, &y.tr, &y.suh, &y.svh));
        }
        assert_eq!(a.plains, b.plains);
        assert_eq!(a.plains32, b.plains32);
        let (am, bm) = (a.moe.as_ref().unwrap(), b.moe.as_ref().unwrap());
        assert_eq!((am.bpb, &am.g_tr, &am.u_tr, &am.d_tr, &am.suh_g, &am.svh_g, &am.suh_u, &am.svh_u, &am.suh_d, &am.svh_d),
                   (bm.bpb, &bm.g_tr, &bm.u_tr, &bm.d_tr, &bm.suh_g, &bm.svh_g, &bm.suh_u, &bm.svh_u, &bm.suh_d, &bm.svh_d));
    }

    #[test]
    fn cache_image_round_trip_is_byte_identical() {
        let dir = std::env::temp_dir().join(format!("nlctest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("layer-00000.bin");
        let s = synthetic();
        let e = write_image(&p, &s).unwrap();
        assert!(e.bytes > 0 && e.sha256.len() == 64);
        let back = read_image(&p, Some(&e), true).unwrap();
        assert_same(&s, &back);
        // verify=off still round-trips
        let back2 = read_image(&p, None, false).unwrap();
        assert_same(&s, &back2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_image_is_refused_when_verify_on() {
        let dir = std::env::temp_dir().join(format!("nlctest-c-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("layer-00001.bin");
        let s = synthetic();
        let e = write_image(&p, &s).unwrap();
        // flip one payload byte
        let mut b = std::fs::read(&p).unwrap();
        let last = b.len() - 1;
        b[last] ^= 0xff;
        std::fs::write(&p, &b).unwrap();
        assert!(read_image(&p, Some(&e), true).is_err(), "verify=on must refuse a corrupted image");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn order_buf_returns_strictly_in_order_under_parallel_producers() {
        let n = 24;
        let buf = std::sync::Arc::new(OrderBuf::new(n));
        let mut hs = Vec::new();
        for t in 0..4usize {
            let b = buf.clone();
            hs.push(std::thread::spawn(move || {
                let mut i = t; // interleaved claims, deliberately out of order
                while i < n {
                    std::thread::sleep(std::time::Duration::from_micros((i % 5) as u64 * 37));
                    b.submit(i, i * 7);
                    i += 4;
                }
            }));
        }
        for h in hs { h.join().unwrap(); }
        for want in 0..n {
            assert_eq!(buf.next(want).unwrap(), Some(want * 7), "item {want} out of order");
        }
        assert!(buf.next(n).unwrap().is_none());
    }

    #[test]
    fn order_buf_error_surfaces_at_the_consumer() {
        let buf = OrderBuf::<usize>::new(4);
        buf.submit(0, 10usize);
        buf.set_err("boom".into());
        assert_eq!(buf.next(0).unwrap(), Some(10)); // already-staged items still deliver
        assert!(buf.next(1).is_err());
    }

    #[test]
    fn claim_gate_bounded_pipeline_completes_under_slow_consumer() {
        // Regression for the LOADFAST deadlock: the look-ahead bound must live on the CLAIM side,
        // never on submit — 4 producers, bound 3, deliberately slow in-order consumer. The old
        // design (bound at submit) hung forever here; claims-in-order cannot.
        let n = 24;
        let gate = std::sync::Arc::new(ClaimGate::new());
        let order = std::sync::Arc::new(OrderBuf::<usize>::new(n));
        let (n_, bound) = (n, 3);
        std::thread::scope(|s| {
            for _ in 0..4 {
                let (gate, order) = (gate.clone(), order.clone());
                s.spawn(move || {
                    while let Some(l) = gate.claim(n_, bound) {
                        std::thread::sleep(std::time::Duration::from_micros((l % 7) as u64 * 13));
                        order.submit(l, l * 3);
                    }
                });
            }
            for want in 0..n {
                let v = order.next(want).unwrap().expect("deadlock: layer never arrived");
                assert_eq!(v, want * 3, "item {want} out of order");
                gate.consumed();
                std::thread::sleep(std::time::Duration::from_micros(200)); // slow uploader
            }
        });
    }
}
