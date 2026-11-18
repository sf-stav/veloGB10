//! The compressor (B.5, exact prefill + decode state machine) and the indexer (B.6, CSA only) of the CPU reference.
//! Siblings: `ops.rs` (primitives), `attn_moe.rs` (the attention that consumes them); `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------
// Compressor (§B.5, model.py:285-383 — exact prefill + decode state machine)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompressorWeights {
    pub wkv: Vec<f32>,   // [coff*d, dim] fp32
    pub wgate: Vec<f32>, // [coff*d, dim] fp32
    pub norm: Vec<f32>,  // [d] fp32
    pub ape: Vec<f32>,   // [ratio, coff*d] fp32 — added to SCORES only
    pub ratio: usize,
    pub head_dim: usize,  // d
    pub rope_dim: usize,  // rd
    pub overlap: bool,    // ratio == 4 (CSA); false for ratio 128 (HCA)
    pub rotate: bool,     // indexer compressor: Hadamard + FP4 sim on full d
    pub sim_group: usize, // 64 (attention fp8-sim) / 32 (indexer fp4-sim)
    pub dim: usize,       // input dim
}

impl CompressorWeights {
    pub fn coff(&self) -> usize {
        1 + self.overlap as usize
    }
}

/// Incremental decode state (`kv_state`/`score_state`, model.py:309-310).
/// With overlap: state rows [0..ratio) = previous block context, [ratio..2·ratio) = current
/// window. Without: [0..ratio) accumulation rows. score_state inits to −inf.
#[derive(Debug, Clone)]
pub struct CompressorState {
    pub kv_state: Vec<f32>,    // [coff*ratio, coff*d] fp32
    pub score_state: Vec<f32>, // [coff*ratio, coff*d] fp32
}

pub struct Compressor {
    pub w: CompressorWeights,
    pub st: CompressorState,
    pub hadamard: Option<Vec<f32>>, // [d,d] scaled — rotate only
}

impl Compressor {
    pub fn new(w: CompressorWeights) -> Self {
        let coff = w.coff();
        let rows = coff * w.ratio;
        let cols = coff * w.head_dim;
        let hadamard = if w.rotate { Some(hadamard_scaled(w.head_dim)) } else { None };
        Compressor {
            st: CompressorState {
                kv_state: vec![0.0; rows * cols],
                score_state: vec![f32::NEG_INFINITY; rows * cols],
            },
            w,
            hadamard,
        }
    }

    /// `Compressor.forward`. x [s, dim] bf16-valued f32. `rope` is the layer's
    /// shared freqs table. `cache` is this compressor's own cache slice
    /// (attention compressor: aliased rows 128.. of the layer kv_cache; indexer:
    /// its own [max_seq//4, 128]). Returns the pooled rows ([nblocks, d] prefill,
    /// [1, d] decode) when a compression fired, else None.
    pub fn forward(
        &mut self,
        x: &[f32],
        s: usize,
        start_pos: usize,
        rope: &RopeTable,
        norm_eps: f32,
        cache: &mut [f32],
    ) -> Option<Vec<f32>> {
        let w = &self.w;
        let (ratio, d, rd, overlap, rotate) = (w.ratio, w.head_dim, w.rope_dim, w.overlap, w.rotate);
        let coff = w.coff();
        let cd = coff * d;
        assert_eq!(x.len(), s * w.dim);
        // compression in fp32 (x values already f32); wkv/wgate are fp32 Linears
        let kv_full = gemm_f32(x, s, w.dim, &w.wkv, cd);
        let score_full = gemm_f32(x, s, w.dim, &w.wgate, cd);
        let mut pooled: Vec<f32>; // [nblocks, d]
        let mut block_positions: Vec<usize>; // first-token position per pooled row
        let mut cache_row0: usize;
        if start_pos == 0 {
            let should_compress = s >= ratio;
            let remainder = s % ratio;
            let cutoff = s - remainder;
            let offset = if overlap { ratio } else { 0 };
            if overlap && cutoff >= ratio {
                // stash the LAST FULL block as overlap context (raw rows, all coff*d dims)
                for j in 0..ratio {
                    let dst = j * cd;
                    let src = (cutoff - ratio + j) * cd;
                    self.st.kv_state[dst..dst + cd].copy_from_slice(&kv_full[src..src + cd]);
                    for c in 0..cd {
                        self.st.score_state[dst + c] = score_full[src + c] + w.ape[j * cd + c];
                    }
                }
            }
            if remainder > 0 {
                for j in 0..remainder {
                    let dst = (offset + j) * cd;
                    let src = (cutoff + j) * cd;
                    self.st.kv_state[dst..dst + cd].copy_from_slice(&kv_full[src..src + cd]);
                    for c in 0..cd {
                        self.st.score_state[dst + c] = score_full[src + c] + w.ape[j * cd + c];
                    }
                }
            }
            let nb = cutoff / ratio;
            pooled = vec![0.0f32; nb * d];
            for b in 0..nb {
                // assemble the pool rows: overlap → 2·ratio rows (prev block dims :d
                // then current block dims d:), else ratio rows.
                let nrow = if overlap { 2 * ratio } else { ratio };
                // per-column softmax over rows
                let mut kvs = vec![0.0f32; nrow * d];
                let mut scs = vec![f32::NEG_INFINITY; nrow * d];
                for j in 0..nrow {
                    for dd in 0..d {
                        let (kv_v, sc_v);
                        if overlap {
                            if j < ratio {
                                if b == 0 {
                                    kv_v = 0.0;
                                    sc_v = f32::NEG_INFINITY;
                                } else {
                                    let src = ((b - 1) * ratio + j) * cd + dd;
                                    kv_v = kv_full[src];
                                    sc_v = score_full[src] + w.ape[j * cd + dd];
                                }
                            } else {
                                let src = (b * ratio + (j - ratio)) * cd + d + dd;
                                kv_v = kv_full[src];
                                sc_v = score_full[src] + w.ape[(j - ratio) * cd + d + dd];
                            }
                        } else {
                            let src = (b * ratio + j) * cd + dd;
                            kv_v = kv_full[src];
                            sc_v = score_full[src] + w.ape[j * cd + dd];
                        }
                        kvs[j * d + dd] = kv_v;
                        scs[j * d + dd] = sc_v;
                    }
                }
                // pooled = (kvs · softmax(scs, dim=rows)).sum(rows), fp32.
                // torch softmax NORMALIZES first (p_j = e_j/z), then Σ p_j·kv_j
                // — that rounding order is load-bearing (do not divide at the end).
                for dd in 0..d {
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..nrow {
                        mx = mx.max(scs[j * d + dd]);
                    }
                    let mut z = 0.0f32;
                    for j in 0..nrow {
                        z += (scs[j * d + dd] - mx).exp();
                    }
                    let mut acc = 0.0f32;
                    for j in 0..nrow {
                        let p = (scs[j * d + dd] - mx).exp() / z;
                        acc += kvs[j * d + dd] * p;
                    }
                    pooled[b * d + dd] = acc;
                }
            }
            block_positions = (0..nb).map(|b| b * ratio).collect();
            cache_row0 = 0;
            if !should_compress {
                return None;
            }
        } else {
            let should_compress = (start_pos + 1) % ratio == 0;
            // score += ape[start_pos % ratio]  (score row for this token only)
            let mut score0 = score_full[..cd].to_vec();
            for c in 0..cd {
                score0[c] += w.ape[(start_pos % ratio) * cd + c];
            }
            pooled = Vec::new();
            if overlap {
                let slot = ratio + start_pos % ratio;
                self.st.kv_state[slot * cd..(slot + 1) * cd].copy_from_slice(&kv_full[..cd]);
                self.st.score_state[slot * cd..(slot + 1) * cd].copy_from_slice(&score0);
                if should_compress {
                    let nrow = 2 * ratio;
                    let mut kvs = vec![0.0f32; nrow * d];
                    let mut scs = vec![0.0f32; nrow * d];
                    for j in 0..nrow {
                        for dd in 0..d {
                            if j < ratio {
                                kvs[j * d + dd] = self.st.kv_state[j * cd + dd];
                                scs[j * d + dd] = self.st.score_state[j * cd + dd];
                            } else {
                                kvs[j * d + dd] = self.st.kv_state[j * cd + d + dd];
                                scs[j * d + dd] = self.st.score_state[j * cd + d + dd];
                            }
                        }
                    }
                    pooled = vec![0.0f32; d];
                    for dd in 0..d {
                        let mut mx = f32::NEG_INFINITY;
                        for j in 0..nrow {
                            mx = mx.max(scs[j * d + dd]);
                        }
                        let mut z = 0.0f32;
                        for j in 0..nrow {
                            z += (scs[j * d + dd] - mx).exp();
                        }
                        let mut acc = 0.0f32;
                        for j in 0..nrow {
                            let p = (scs[j * d + dd] - mx).exp() / z;
                            acc += kvs[j * d + dd] * p;
                        }
                        pooled[dd] = acc;
                    }
                    // shift: state rows [0..ratio) <- [ratio..2ratio)
                    for j in 0..ratio {
                        let dst = j * cd;
                        let src = (ratio + j) * cd;
                        let (a, b) = self.st.kv_state.split_at_mut(src);
                        a[dst..dst + cd].copy_from_slice(&b[..cd]);
                        let (a, b) = self.st.score_state.split_at_mut(src);
                        a[dst..dst + cd].copy_from_slice(&b[..cd]);
                    }
                }
            } else {
                let slot = start_pos % ratio;
                self.st.kv_state[slot * cd..(slot + 1) * cd].copy_from_slice(&kv_full[..cd]);
                self.st.score_state[slot * cd..(slot + 1) * cd].copy_from_slice(&score0);
                if should_compress {
                    pooled = vec![0.0f32; d];
                    for dd in 0..d {
                        let mut mx = f32::NEG_INFINITY;
                        for j in 0..ratio {
                            mx = mx.max(self.st.score_state[j * cd + dd]);
                        }
                        let mut z = 0.0f32;
                        for j in 0..ratio {
                            z += (self.st.score_state[j * cd + dd] - mx).exp();
                        }
                        let mut acc = 0.0f32;
                        for j in 0..ratio {
                            let p = (self.st.score_state[j * cd + dd] - mx).exp() / z;
                            acc += self.st.kv_state[j * cd + dd] * p;
                        }
                        pooled[dd] = acc;
                    }
                }
            }
            if !should_compress {
                return None;
            }
            block_positions = vec![start_pos + 1 - ratio];
            cache_row0 = start_pos / ratio;
        }
        // post-pooling: bf16 -> RMSNorm -> RoPE at first-token position -> QAT-sim
        let nrows = pooled.len() / d;
        for (b, pos) in block_positions.iter().enumerate().take(nrows) {
            let row = &mut pooled[b * d..(b + 1) * d];
            round_bf16(row);
            rms_norm_row(row, &w.norm, norm_eps);
            apply_rope(row, 1, d, rope, &[*pos], false);
            if rotate {
                let h = self.hadamard.as_ref().unwrap();
                rotate_activation(row, 1, d, d, h);
                fp4_act_quant_sim(row, 1, d, w.sim_group);
            } else {
                act_quant_sim(&mut row[..d - rd], 1, d - rd, w.sim_group);
            }
            cache[(cache_row0 + b) * d..(cache_row0 + b + 1) * d].copy_from_slice(row);
        }
        Some(pooled)
    }
}

// ---------------------------------------------------------------------------
// Indexer (§B.6, model.py:386-439 — CSA only)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct IndexerWeights {
    pub wq_b: Vec<f32>,         // [64*128, q_lora_rank] fp8-dequant
    pub weights_proj: Vec<f32>, // [64, dim] bf16-valued
    pub compressor: CompressorWeights, // ratio 4, d 128, rotate = true
}

pub struct IndexerState {
    pub compressor: Compressor,
    pub kv_cache: Vec<f32>, // [max_seq//4, 128] bf16-valued
}

/// `Indexer.forward` → per-token compress-block index lists (with `offset`
/// already added; −1 = masked). `qr` is the attention's q_lora latent [s, 1024]
/// (bf16-valued). `rope` is the layer table. `hadamard` is the 128×128 scaled
/// rotation. Deterministic top-k per §12.B.2.
#[allow(clippy::too_many_arguments)]
pub fn indexer_forward(
    w: &IndexerWeights,
    st: &mut IndexerState,
    x: &[f32],
    s: usize,
    qr: &[f32],
    start_pos: usize,
    offset: usize,
    rope: &RopeTable,
    hadamard: &[f32],
    cfg: &Dsv4Config,
) -> Vec<Vec<i64>> {
    let dim = cfg.dim;
    let nh = cfg.index_n_heads;
    let hd = cfg.index_head_dim;
    let rd = cfg.rope_head_dim;
    let ratio = 4usize;
    let end_pos = start_pos + s;
    // q = wq_b(qr) -> [s, 64, 128]; RoPE last 64; Hadamard; FP4 sim
    let mut q = quant_gemm(qr, s, cfg.q_lora_rank, &w.wq_b, nh * hd, 128);
    {
        let rows = s * nh;
        let pos: Vec<usize> = (0..rows).map(|i| start_pos + i / nh).collect();
        apply_rope(&mut q, rows, hd, rope, &pos, false);
        rotate_activation(&mut q, rows, hd, hd, hadamard);
        fp4_act_quant_sim(&mut q, rows, hd, 32);
    }
    // own compressor updates the indexer kv cache
    st.compressor.forward(x, s, start_pos, rope, cfg.norm_eps, &mut st.kv_cache);
    // head weights: weights_proj(x) · (128^-0.5 · 64^-0.5), bf16
    let mut weights = gemm_bf16(x, s, dim, &w.weights_proj, nh);
    let wscale = ((hd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
    for v in weights.iter_mut() {
        *v = bf(*v * wscale);
    }
    let nblocks = end_pos / ratio;
    let k = cfg.index_topk.min(nblocks);
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(s);
    for i in 0..s {
        // index_score = einsum(q, kv_cache[:nblocks]); relu; ×weights; sum over heads
        // (bf16 tensor semantics: einsum out bf16, relu exact, mul bf16, sum f32-acc→bf16)
        let mut score = vec![0.0f32; nblocks];
        for t in 0..nblocks {
            let kvrow = &st.kv_cache[t * hd..(t + 1) * hd];
            let mut acc = 0.0f32;
            for h in 0..nh {
                let qrow = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                let dot = bf(dot8(qrow, kvrow)); // einsum output is bf16
                let rel = if dot > 0.0 { dot } else { 0.0 }; // relu_ (bf16, exact)
                acc += bf(rel * weights[i * nh + h]); // bf16 mul
            }
            score[t] = bf(acc);
        }
        if start_pos == 0 {
            // block-causal: token i sees block t ⟺ t < (i+1)//ratio
            let lim = (i + 1) / ratio;
            for (t, v) in score.iter_mut().enumerate() {
                if t >= lim {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let mut idx = topk_deterministic(&score, k);
        if start_pos == 0 {
            let lim = (i + 1) / ratio;
            for v in idx.iter_mut() {
                if *v >= lim as i64 {
                    *v = -1;
                } else {
                    *v += offset as i64;
                }
            }
        } else {
            for v in idx.iter_mut() {
                *v += offset as i64;
            }
        }
        out.push(idx);
    }
    out
}

