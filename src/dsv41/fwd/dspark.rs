//! SL: the DSpark stages (layer slots 40-42) on the GPU, gated per op against leg OD's oracle traces
//! (`--dsv41-kernel-fixture <OD trace dir per fixture>`, written by `--probe-dsv41-oracle --dsv41-oracle-part
//! dspark --dsv41-oracle-trace DIR`): REFb dumps only the drafts of the stages, so OD's oracle (equal to
//! DeepSeek's DSpark modules per op, and to REFb's drafts within the reference's own floor) provides each
//! op's input and output. Every op is fed the TRACE's input (never a chain):
//!
//! | point | input (trace) | ours | reference (trace) |
//! |---|---|---|---|
//! | `ds_attn_mix` / `ds_attn_x` | `hc_in`, `pre_mix_in` | hc mixes; hc_pre + attn_norm | `attn_mix` / `attn_x` |
//! | `ds_q` / `ds_kv` | `attn_x` | wq_a, q_norm, wq_b, RoPE (p+1..p+5); wkv, kv_norm, RoPE, FP8 | `attn_q` / `attn_kv` |
//! | `ds_attn_out` | `attn_x`, the step's window ring `attn_ring` | KA's attention: window to the main position + the block's 5 keys, wo | `attn_out` |
//! | `ds_hc_mid` | `attn_out`, `hc_in` | hc_post | `hc_mid` |
//! | `ds_ffn_mix` / `ds_ffn_x` | `hc_mid`, `attn_mix` (pre) | hc mixes; hc_pre + ffn_norm | `ffn_mix` / `ffn_x` |
//! | `ds_router` / `ds_ffn_out` | `ffn_x` (+ the trace's routing) | 128-expert top-3 router; routed + shared experts | `router_ids` / `ffn_out` |
//! | `ds_hc_out` | `ffn_out`, `hc_mid` | hc_post | `hc_out` |
//! | `ds_main_x` (stage 0) | REFb's `G.dspark_main_hidden` (the prompt's last rows) | main_proj + main_norm | REFb's `G.dspark_main_x` |
//!
//! Not here: the head of the last stage (tied head, Markov chain, confidence) and the draft tokens (leg DS).

use super::cmp::{self, Cmp};
use super::dev::Buf;
use super::gate::{Fix, Gate};
use super::layer::{fmt_of, HC_DIM};
use super::receipts;
use super::weights::LayerW;
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::dump::{Dump, DumpLayer};
use crate::dsv41::gpu::{abi, attn, kv};
use crate::dsv41::weights::DevPtr;
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

pub struct DsFix { pub name: String, pub d: Dump, pub p: usize }

impl DsFix {
    pub fn open(dir: &std::path::Path) -> Result<DsFix> {
        let d = Dump::open(dir)?;
        let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let pos = d.meta.get("positions").and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| Some(v.to_string()))).unwrap_or_default();
        let p: usize = pos.trim_matches(|c| c == '[' || c == ']').split(',').next().and_then(|s| s.trim().parse().ok()).context("trace positions")?;
        Ok(DsFix { name, d, p })
    }
    fn f(&self, l: usize, point: &str) -> Result<Vec<f32>> { self.d.read(DumpLayer::Layer(l as u16), point)?.to_f32() }
}

/// E4M3 code of an exactly representable value (the trace's ring holds quant-dequant values).
fn e4m3_lut() -> HashMap<u32, u8> {
    let mut m = HashMap::new();
    for c in 0u8..=255 { let v = abi::e4m3(c); if v.is_finite() { m.entry(v.to_bits()).or_insert(c); } }
    m
}
/// act_quant of one 512-row of quant-dequant values back to (E4M3 codes, E8M0 per 32): the value-level identity.
fn requant_row(v: &[f32], lut: &HashMap<u32, u8>, q: &mut [u8], s: &mut [u8]) -> Result<()> {
    for b in 0..v.len() / 32 {
        let blk = &v[b * 32..(b + 1) * 32];
        let amax = blk.iter().fold(0f32, |a, x| a.max(x.abs())).max(1e-4);
        let t = amax * (1.0f32 / 448.0);
        let bits = t.to_bits();
        let e = ((bits >> 23) & 0xFF) as i32 - 127 + i32::from(bits & 0x7F_FFFF != 0);
        s[b] = (e + 127) as u8;
        let sc = f32::from_bits(((e + 127) as u32) << 23);
        for (i, x) in blk.iter().enumerate() {
            let y = *x / sc;
            q[b * 32 + i] = *lut.get(&y.to_bits()).with_context(|| format!("ring value {x} / {sc} is not an E4M3 value"))?;
        }
    }
    Ok(())
}

impl<'a> Gate<'a> {
    fn ds_push(&mut self, fx: &str, l: usize, point: &'static str, c: Cmp, extra: String) {
        let t = cmp::tol(self.pass, point, l);
        let pass = if t.rel.is_infinite() { None } else { Some(c.rel <= t.rel && c.same >= t.same) };
        self.rows.push(super::gate::Row { fixture: fx.to_string(), layer: l, point, c, extra, tol: t, pass });
    }

    /// Every per-op check of DSpark stage `w.layer` on one trace; `refb` = the same fixture's REFb dump (main_x).
    pub fn dspark_stage(&mut self, w: &LayerW, fx: &DsFix, refb: Option<&Fix>, cfg: &crate::dsv41::config::Dsv41Config) -> Result<()> {
        let (l, d, name) = (w.layer, cfg.hidden, fx.name.clone());
        let g = self.f.g;
        g.sync()?;
        self.f.b.reset();
        self.keep = super::gate::Keep::default();
        let bk = cfg.dspark_block;
        let hc_in = fx.f(l, "hc_in")?;
        let rows = hc_in.len() / HC_DIM;
        let steps = rows / bk;
        ensure!(steps * bk == rows && steps > 0, "{}: trace rows {rows} are not steps x {bk}", fx.name);
        let pos: Vec<i32> = fx.d.read(DumpLayer::Head, "positions")?.to_f32()?.iter().map(|&x| x as i32).collect();
        ensure!(pos.len() == rows, "{}: {} positions for {rows} rows", fx.name, pos.len());
        let dpos = self.keep_i32(&pos)?;
        let wend: Vec<i32> = (0..rows).map(|r| (fx.p + r / bk) as i32).collect();
        let dwend = self.keep_i32(&wend)?;
        // ---- mixes, attn_x
        let dh = self.keep_bf16(&hc_in)?;
        let rec_a = self.f.mix(w, true, dh, rows)?;
        let am = fx.f(l, "attn_mix")?;
        self.ds_push(&name, l, "ds_attn_mix", cmp::cmp(&g.down::<f32>(rec_a, rows * 24)?, &am), String::new());
        let pre = self.keep_f32(&fx.f(l, "pre_mix_in")?)?;
        let ax = self.f.hc_pre_norm(dh, pre, 4, w.param(C::AttnNorm)?, rows)?;
        let ax_ref = fx.f(l, "attn_x")?;
        self.ds_push(&name, l, "ds_attn_x", cmp::cmp(&g.down_bf16(ax, rows * d)?, &ax_ref), String::new());
        // ---- q and the block's keys on the trace's attn_x
        let dax = self.keep_bf16(&ax_ref)?;
        let (_qr, q) = self.f.q_path(w, dax, dpos, rows, false)?;
        let (h, hd) = (cfg.heads, cfg.head_dim);
        self.ds_push(&name, l, "ds_q", cmp::cmp(&g.down_bf16(q, rows * h * hd)?, &fx.f(l, "attn_q")?), "after RoPE".into());
        let raw = self.f.b.take(rows * hd * 2)?;
        self.f.lin_rows("ds.attn.wkv", w, C::AttnWkv, None, dax, d, d, rows, raw, hd, false)?;
        let (bq, bs) = (g.buf(rows * hd)?, g.buf(rows * hd / 32)?);
        let norm = w.param(C::AttnKvNorm)?;
        let freq = self.f.b.take(128)?;
        g.h2d_async(freq, &self.f.rope.kc.plain)?;
        for st in 0..steps {
            let a = kv::SwaArgs { kv_raw: raw + (st * bk * hd * 2) as u64, norm_w: norm.ptr.0, inv_freq: freq, ring_q: 0, ring_s: 0,
                                  chunk_q: bq.ptr() + (st * bk * hd) as u64, chunk_s: bs.ptr() + (st * bk * hd / 32) as u64,
                                  p0: pos[st * bk] as i64, rows: bk as i32, norm_fmt: fmt_of(norm)?, eps: cfg.rms_eps as f32 };
            g.launch1("d41_kc_swa_write", (bk as u32, 1, 1), (128, 1, 1), 0, &a)?;
        }
        receipts::kernel("ds.kv_block", "d41_kc_swa_write", bk);
        let ours_kv = cmp::decode_fp8_rows(&g.down::<u8>(bq.ptr(), rows * hd)?, &g.down::<u8>(bs.ptr(), rows * hd / 32)?, rows, hd);
        self.ds_push(&name, l, "ds_kv", cmp::cmp(&ours_kv, &fx.f(l, "attn_kv")?), "block rows, FP8".into());
        // ---- attention per step: the trace's window ring (requantised to FP8 planes) + the block's 5 keys
        let ring = fx.f(l, "attn_ring")?;
        let win = cfg.window;
        ensure!(ring.len() == steps * win * hd, "{}: attn_ring holds {} values", fx.name, ring.len());
        let lut = e4m3_lut();
        let (lq, ls) = (g.buf(steps * win * hd)?, g.buf(steps * win * hd / 32)?);
        let mut hq = vec![0u8; steps * win * hd];
        let mut hs = vec![0u8; steps * win * hd / 32];
        for st in 0..steps {
            let p = fx.p + st;
            let base = p.saturating_sub(win - 1);
            for (j, t) in (base..=p).enumerate() {
                let slot = t % win;
                let src = &ring[(st * win + slot) * hd..(st * win + slot + 1) * hd];
                let o = st * win + j;
                requant_row(src, &lut, &mut hq[o * hd..(o + 1) * hd], &mut hs[o * hd / 32..(o + 1) * hd / 32])?;
            }
        }
        g.h2d(lq.ptr(), &hq)?;
        g.h2d(ls.ptr(), &hs)?;
        let dummy: Buf = g.buf(cfg.swa_slots() * hd)?;
        let dummy_s: Buf = g.buf(cfg.swa_slots() * hd / 32)?;
        let o = self.f.b.take(rows * h * hd * 2)?;
        let err = self.f.b.take(4)?;
        g.memset(err, 4, 0)?;
        for st in 0..steps {
            let p = fx.p + st;
            let base = p.saturating_sub(win - 1);
            let lane = attn::LaneSrc { swa_q: dummy.p(), swa_s: dummy_s.p(), slots: cfg.swa_slots(),
                                       lin: Some((DevPtr(lq.ptr() + (st * win * hd) as u64), DevPtr(ls.ptr() + (st * win * hd / 32) as u64), base as i64)),
                                       kv: None, ext: Some((DevPtr(bq.ptr() + (st * bk * hd) as u64), DevPtr(bs.ptr() + (st * bk * hd / 32) as u64), bk)) };
            let r0 = st * bk;
            let call = attn::AttnCall { rows: bk, heads: h, q: DevPtr(q + (r0 * h * hd * 2) as u64), q_stride: h * hd, sink: w.param(C::AttnSink)?.ptr,
                                        pos: DevPtr(dpos + (r0 * 4) as u64), win_end: DevPtr(dwend + (r0 * 4) as u64), lane: DevPtr(0), sel: DevPtr(0), sel_stride: 0,
                                        sel_cap: 0, n_sel: DevPtr(0), lanes: vec![lane], out: DevPtr(o + (r0 * h * hd * 2) as u64), out_stride: h * hd,
                                        out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ, rope_freq: self.f.rope.ka.plain, err: DevPtr(err) };
            attn::launch_prefill(g.l(), &call)?;
        }
        receipts::kernel("ds.attn (window+block keys)", "d41_attn_prefill", bk);
        ensure!(g.down::<i32>(err, 1)?[0] == 0, "DSpark attention error word");
        let ao = self.f.wo(w, o, rows)?;
        let ao_ref = fx.f(l, "attn_out")?;
        self.ds_push(&name, l, "ds_attn_out", cmp::cmp(&g.down_bf16(ao, rows * d)?, &ao_ref), format!("{steps} steps x {bk} rows"));
        // ---- hc_mid, the FFN side, hc_out
        let dao = self.keep_bf16(&ao_ref)?;
        let hm = self.f.hc_post(dao, dh, rec_a, rows)?;
        let hm_ref = fx.f(l, "hc_mid")?;
        self.ds_push(&name, l, "ds_hc_mid", cmp::cmp(&g.down_bf16(hm, rows * HC_DIM)?, &hm_ref), String::new());
        let dhm = self.keep_bf16(&hm_ref)?;
        let rec_f = self.f.mix(w, false, dhm, rows)?;
        self.ds_push(&name, l, "ds_ffn_mix", cmp::cmp(&g.down::<f32>(rec_f, rows * 24)?, &fx.f(l, "ffn_mix")?), String::new());
        let apre = self.keep_f32(&am)?;
        let fxv = self.f.hc_pre_norm(dhm, apre, 24, w.param(C::FfnNorm)?, rows)?;
        let fx_ref = fx.f(l, "ffn_x")?;
        self.ds_push(&name, l, "ds_ffn_x", cmp::cmp(&g.down_bf16(fxv, rows * d)?, &fx_ref), String::new());
        let dfx = self.keep_bf16(&fx_ref)?;
        let (ids, _wts) = self.f.router(w, dfx, rows)?;
        let (_, k) = cfg.moe_of(l);
        let gi = g.down::<i32>(ids, rows * k)?;
        let rid = fx.f(l, "router_ids")?;
        let mut same = 0usize;
        for r in 0..rows {
            let mut a: Vec<i64> = gi[r * k..(r + 1) * k].iter().map(|&x| x as i64).collect();
            let mut b: Vec<i64> = rid[r * k..(r + 1) * k].iter().map(|&x| x as i64).collect();
            a.sort_unstable(); b.sort_unstable();
            if a == b { same += 1; }
        }
        self.ds_push(&name, l, "ds_router", Cmp { rel: 0.0, same: same as f64 / rows as f64, max_abs: 0.0, n: rows }, format!("{same}/{rows} rows same top-{k} set"));
        let dids = self.keep_i32(&rid.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
        let dw = self.keep_f32(&fx.f(l, "router_w")?)?;
        let fo = self.f.ffn(w, dfx, rows, dids, dw)?;
        let fo_ref = fx.f(l, "ffn_out")?;
        self.ds_push(&name, l, "ds_ffn_out", cmp::cmp(&g.down_bf16(fo, rows * d)?, &fo_ref), "trace routing".into());
        let dfo = self.keep_bf16(&fo_ref)?;
        let ho = self.f.hc_post(dfo, dhm, rec_f, rows)?;
        self.ds_push(&name, l, "ds_hc_out", cmp::cmp(&g.down_bf16(ho, rows * HC_DIM)?, &fx.f(l, "hc_out")?), String::new());
        // ---- stage 0: main_x of the prompt's last rows against REFb directly
        if w.lin.contains_key(&(C::DsMainProj, None)) {
            if let Some(rf) = refb {
                let mh = rf.d.global("dspark_main_hidden")?;
                let mx = rf.d.global("dspark_main_x")?;
                let n = mh.n_rows();
                let kin = mh.stride();
                let dmh = self.keep_bf16(&mh.f)?;
                let y = self.f.b.take(n * d * 2)?;
                self.f.lin_rows("ds.main_proj", w, C::DsMainProj, None, dmh, kin, kin, n, y, d, false)?;
                let nw = w.param(C::DsMainNorm)?;
                let z = self.f.b.take(n * d * 2)?;
                let a = crate::dsv41::gpu::mixers::RmsNormArgs { x: y, x_ld: d as i64, w: nw.ptr.0, y: z, y_ld: d as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: n as i32,
                                                                 eps: cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
                g.launch1("d41_rmsnorm", (n as u32, 1, 1), (256, 1, 1), 0, &a)?;
                self.ds_push(&name, l, "ds_main_x", cmp::cmp(&g.down_bf16(z, n * d)?, &mx.f), format!("{n} prompt rows vs REFb"));
            }
        }
        g.sync()
    }

    fn keep_bf16(&mut self, v: &[f32]) -> Result<u64> {
        let b: Vec<u16> = v.iter().map(|&x| cmp::f32_to_bf16_bits(x)).collect();
        self.keep.bytes(self.f.g, super::layer::bytemuck_u16(&b))
    }
    fn keep_f32(&mut self, v: &[f32]) -> Result<u64> { self.keep.bytes(self.f.g, super::layer::bytemuck_f32(v)) }
    fn keep_i32(&mut self, v: &[i32]) -> Result<u64> { self.keep.bytes(self.f.g, super::layer::bytemuck_i32(v)) }
}
