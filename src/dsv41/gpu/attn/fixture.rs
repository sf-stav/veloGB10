//! The sparse attention core against REFb's reference dumps (leg KA), through a fixture written by
//! `tools/dsv41_ref/ka_attn_fixture.py build` (dsv41 dump format): per layer class, the real queries, the
//! real window KV (FP8 planes) and compressed KV (288-B records, re-encoded bit-exact from REFb's values),
//! REFb's selections and the reference outputs (REFb's emulated sparse_attn in the reference's own prefill
//! index layout, then the inverse RoPE: `attno`; then act_quant: `woain`).
//!
//! For every layer: the prefill entry with no epilogue against an exact f64 attention on the same caches,
//! with the inverse RoPE against `attno`, with RoPE + act_quant against `woain`; rows in runs of six as
//! verify groups (one lane each, the window from a 134-slot ring) through the decode entry, bitwise equal to
//! the prefill rows; the host RoPE frequencies (`rope_inv_freq_for_layer`) against the reference's.
//! The kernel's WO-A input is written to `<fixture>/ka_kernel_out` (point `woaink`) for
//! `ka_attn_fixture.py check`, which applies WO-A / WO-B and compares with REFb's attn_out.

use super::probe::{run_case, Case, HostLane, WinSrc};
use super::reference as cref;
use super::{EPI_AQ, EPI_ROPE};
use crate::dsv41::dump::{Dump, DumpLayer, DumpTensor, DumpWriter};
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use std::path::Path;
use std::sync::Arc;

fn as_u16(t: &DumpTensor) -> Vec<u16> { t.bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect() }
fn as_i32(t: &DumpTensor) -> Vec<i32> { t.bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() }
fn as_f32(t: &DumpTensor) -> Vec<f32> { t.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() }

/// rel-L2 of two bf16 arrays and the fraction of bit-identical elements.
fn cmp_bf16(a: &[u16], b: &[u16]) -> (f64, f64) {
    let af: Vec<f32> = a.iter().map(|x| cref::bf16_f32(*x)).collect();
    let bf: Vec<f32> = b.iter().map(|x| cref::bf16_f32(*x)).collect();
    let same = a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len().max(1) as f64;
    (cref::rel_l2_f32(&af, &bf), same)
}

pub fn run(dev: &Arc<CudaDevice>, dir: &Path) -> Result<Vec<String>> {
    let d = Dump::open(dir).with_context(|| format!("fixture {}", dir.display()))?;
    let extra = d.meta.get("extra").context("fixture: no extra")?;
    let p_total = extra.get("P").and_then(|v| v.as_u64()).context("fixture: P")? as usize;
    let layers = extra.get("layers").and_then(|v| v.as_object()).context("fixture: layers")?;
    let mut lines = vec![format!("fixture {} (P = {p_total}, source {}):", dir.display(), extra.get("source_dump").and_then(|v| v.as_str()).unwrap_or("?"))];
    let mut out = DumpWriter::create(&dir.join("ka_kernel_out"), "gb10_inference --probe-dsv41-kernel --dsv41-kernel-set attn", "", "native", "exact", &[], (0, p_total as u64))?;
    out.note("woaink", "the kernel's WO-A input (prefill entry, inverse RoPE + act_quant), rows = the fixture's rows");
    let mut keys: Vec<(usize, &serde_json::Value)> = layers.iter().filter_map(|(k, v)| k.parse().ok().map(|l| (l, v))).collect();
    keys.sort_by_key(|x| x.0);
    let mut fail = Vec::new();
    for (layer, info) in keys {
        let lay = DumpLayer::Layer(layer as u16);
        let rd = |p: &str| d.read(lay, p).with_context(|| format!("layer {layer}: point {p}"));
        let q = as_u16(&rd("q")?);
        let rows = as_i32(&rd("rows")?);
        let kvq = rd("kvq")?.bytes;
        let kvs = rd("kvs")?.bytes;
        let comprec = rd("comprec")?.bytes;
        let sel = as_i32(&rd("sel")?);
        let nsel = as_i32(&rd("nsel")?);
        let sink = as_f32(&rd("sink")?);
        let attno = as_u16(&rd("attno")?);
        let woain = as_u16(&rd("woain")?);
        let ratio = info.get("ratio").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let bits: Vec<u32> = info.get("rope_freq_bits").and_then(|v| v.as_array()).context("rope_freq_bits")?.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect();
        ensure!(bits.len() == 32, "layer {layer}: rope_freq_bits");
        let mut freq = [0f32; 32];
        for (i, b) in bits.iter().enumerate() { freq[i] = f32::from_bits(*b); }
        let theta = info.get("theta").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let orig = info.get("yarn_original_seq_len").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let host = if ratio > 0 { super::rope_inv_freq(theta, Some((16.0, orig, 32.0, 1.0)))? } else { super::rope_inv_freq(theta, None)? };
        let ulp = host.iter().zip(freq.iter()).map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).abs()).max().unwrap_or(0);
        let s = rows.len();
        let pw = kvq.len() / 512;
        // the host truth of the caches
        let mut lane = HostLane { maxpos: pw - 1, win_q: kvq.clone(), win_s: kvs.clone(), win: Vec::with_capacity(pw), kv: vec![], comp: vec![], ext_q: vec![], ext_s: vec![], ext: vec![] };
        for p in 0..pw { lane.win.push(cref::fp8_row_decode(&kvq[p * 512..p * 512 + 512], &kvs[p * 16..p * 16 + 16])); }
        if ratio > 0 {
            lane.kv = comprec.clone();
            for r in comprec.chunks_exact(288) { lane.comp.push(cref::kv_record_decode(r)); }
        }
        let case = Case { heads: 64, pos: rows.clone(), wend: rows.clone(), lane: vec![0; s], sel: sel.clone(), nsel: nsel.clone(), sel_cap: 512,
                          q: q.clone(), sink: sink.clone(), use_kv: ratio > 0, use_ext: false, win: vec![WinSrc::Lin { base: 0 }], freq };
        let lanes = vec![lane];
        let raw = run_case(dev, &lanes, &case, false, 0)?;
        ensure!(raw.err == 0, "layer {layer}: kernel error word {}", raw.err);
        let scale = super::softmax_scale();
        let (mut w64, mut m64) = (0f64, 0f64);
        for r in 0..s {
            let qr = case.qrow(r);
            let k = case.keys(&lanes, r);
            let f = cref::attend_f64(&qr, 64, &k, &sink, scale);
            let got: Vec<f32> = raw.o[r * 64 * 512..(r + 1) * 64 * 512].iter().map(|b| cref::bf16_f32(*b)).collect();
            let e = cref::rel_l2(&got, &f);
            w64 = w64.max(e);
            m64 += e / s as f64;
        }
        let ro = run_case(dev, &lanes, &case, false, EPI_ROPE)?;
        let (r_attno, same_attno) = cmp_bf16(&ro.o, &attno);
        let wo = run_case(dev, &lanes, &case, false, EPI_ROPE | EPI_AQ)?;
        let (r_woain, same_woain) = cmp_bf16(&wo.o, &woain);
        let wbytes: Vec<u8> = wo.o.iter().flat_map(|x| x.to_le_bytes()).collect();
        out.put(lay, "woaink", "BF16", &[s as u64, 64 * 512], &wbytes)?;
        // verify groups: runs of six consecutive rows, one lane each with a 134-slot ring, decode entry
        let mut groups: Vec<usize> = Vec::new();
        let mut i = 0;
        while i + 6 <= s && groups.len() < 4 {
            if (0..6).all(|j| rows[i + j] == rows[i] + j as i32) && rows[i] >= 0 { groups.push(i); i += 6; } else { i += 1; }
        }
        let mut dec_same = true;
        if !groups.is_empty() {
            let mut gl = Vec::new();
            let mut gc = Case { heads: 64, pos: vec![], wend: vec![], lane: vec![], sel: vec![], nsel: vec![], sel_cap: 512, q: vec![], sink: sink.clone(),
                                use_kv: ratio > 0, use_ext: false, win: vec![], freq };
            for (gi, &g0) in groups.iter().enumerate() {
                let l0 = &lanes[0];
                gl.push(HostLane { maxpos: l0.maxpos, win_q: l0.win_q.clone(), win_s: l0.win_s.clone(), win: l0.win.clone(), kv: l0.kv.clone(), comp: l0.comp.clone(),
                                   ext_q: vec![], ext_s: vec![], ext: vec![] });
                gc.win.push(WinSrc::Ring { last: rows[g0 + 5] as usize });
                for j in g0..g0 + 6 {
                    gc.pos.push(rows[j]); gc.wend.push(rows[j]); gc.lane.push(gi as i32); gc.nsel.push(nsel[j]);
                    gc.sel.extend_from_slice(&sel[j * 512..(j + 1) * 512]);
                    gc.q.extend_from_slice(&q[j * 64 * 512..(j + 1) * 64 * 512]);
                }
            }
            let dv = run_case(dev, &gl, &gc, true, EPI_ROPE | EPI_AQ)?;
            for (gi, &g0) in groups.iter().enumerate() {
                for j in 0..6 {
                    let a = &dv.o[(gi * 6 + j) * 64 * 512..(gi * 6 + j + 1) * 64 * 512];
                    let b = &wo.o[(g0 + j) * 64 * 512..(g0 + j + 1) * 64 * 512];
                    dec_same &= a == b;
                }
            }
        }
        let ok = w64 < 6e-3 && r_attno < 6e-3 && r_woain < 1.5e-2 && dec_same;
        if !ok { fail.push(layer); }
        lines.push(format!("  layer {layer:2} (ratio {ratio}, {s} rows, {} selected per row): rel-L2 vs f64 worst {w64:.2e} mean {m64:.2e} | vs REFb attn_o {r_attno:.2e} ({:.1} % bf16-identical) | vs REFb WO-A input {r_woain:.2e} ({:.1} %) | {} verify rows decode == prefill: {} | host RoPE freqs vs reference: max {ulp} ulp{}",
                           nsel.iter().max().unwrap_or(&0), 100.0 * same_attno, 100.0 * same_woain, groups.len() * 6, if dec_same { "yes" } else { "NO" },
                           if ok { "" } else { "  FAIL" }));
    }
    let p = out.finish()?;
    lines.push(format!("  kernel WO-A inputs written: {} (end-to-end: tools/dsv41_ref/ka_attn_fixture.py check --fixture {} --kernel-out {})", p.display(), dir.display(), dir.join("ka_kernel_out").display()));
    ensure!(fail.is_empty(), "fixture {}: layers {fail:?} out of bounds:\n{}", dir.display(), lines.join("\n"));
    Ok(lines)
}
