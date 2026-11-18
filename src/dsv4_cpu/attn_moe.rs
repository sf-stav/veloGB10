//! The attention (B.1-B.4) and MoE (B.9) forwards of the CPU reference.
//! Siblings: `compressor.rs` (compressor / indexer), `block.rs` (composes these into a block); `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------
// Attention (§B.1–B.4, model.py:442-548)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AttnWeights {
    pub wq_a: Vec<f32>,     // [q_lora_rank, dim] fp8-dequant
    pub q_norm: Vec<f32>,   // [q_lora_rank]
    pub wq_b: Vec<f32>,     // [64*512, q_lora_rank] fp8-dequant
    pub wkv: Vec<f32>,      // [512, dim] fp8-dequant
    pub kv_norm: Vec<f32>,  // [512]
    pub sink: Vec<f32>,     // [64] fp32 — denominator only
    pub wo_a: Vec<f32>,     // [8*1024, 4096] bf16-valued (§F.2 dequant)
    pub wo_b: Vec<f32>,     // [4096, 8192] fp8-dequant
    pub compressor: Option<CompressorWeights>,
    pub indexer: Option<IndexerWeights>,
    pub kind: LayerKind,
}

pub struct AttnState {
    pub kv_cache: Vec<f32>, // [cache_rows, 512] bf16-valued
    pub cache_rows: usize,
    pub compressor: Option<Compressor>,
    pub indexer: Option<IndexerState>,
}

impl AttnState {
    pub fn new(cfg: &Dsv4Config, w: &AttnWeights, max_seq_len: usize) -> Self {
        let win = cfg.window_size;
        let cache_rows = match w.kind {
            LayerKind::Swa => win,
            LayerKind::Csa => win + max_seq_len / 4,
            LayerKind::Hca => win + max_seq_len / 128,
        };
        let compressor = w.compressor.as_ref().map(|cw| Compressor::new(cw.clone()));
        let indexer = w.indexer.as_ref().map(|iw| IndexerState {
            compressor: Compressor::new(iw.compressor.clone()),
            kv_cache: vec![0.0; (max_seq_len / 4) * cfg.index_head_dim],
        });
        AttnState {
            kv_cache: vec![0.0; cache_rows * cfg.head_dim],
            cache_rows,
            compressor,
            indexer,
        }
    }
}

/// Q/KV common sub-path (§B.1.1–2). Returns (qr [s,1024], q [s,64,512], kv [s,512]).
/// All bf16-valued f32. `x` [s, dim] is the attn_norm output.
pub fn attn_qkv(
    w: &AttnWeights,
    x: &[f32],
    s: usize,
    start_pos: usize,
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = cfg.dim;
    let qlr = cfg.q_lora_rank;
    let nh = cfg.n_heads;
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    // qr = q_norm(wq_a(x))   [s, 1024]
    let qr_pre = quant_gemm(x, s, dim, &w.wq_a, qlr, 128);
    let qr = rms_norm(&qr_pre, s, qlr, &w.q_norm, cfg.norm_eps);
    // q = wq_b(qr) -> [s, 64, 512]; weight-free per-head RMS rescale (bf16 per-op)
    let mut q = quant_gemm(&qr, s, qlr, &w.wq_b, nh * hd, 128);
    for i in 0..s * nh {
        let row = &mut q[i * hd..(i + 1) * hd];
        // q.square().mean(-1) + eps -> rsqrt -> q *= — torch bf16 op-by-op:
        // square is a bf16 tensor (per-op rounding), mean reduces f32 (tree)
        let mut sq: Vec<f32> = row.iter().map(|&v| bf(v * v)).collect();
        let ss = pairwise_sum(&mut sq);
        let mean = bf(ss / hd as f32);
        let arg = bf(mean + cfg.norm_eps);
        let r = bf(arg.sqrt().recip());
        for v in row.iter_mut() {
            *v = bf(*v * r);
        }
    }
    {
        let rows = s * nh;
        let pos: Vec<usize> = (0..rows).map(|i| start_pos + i / nh).collect();
        apply_rope(&mut q, rows, hd, rope, &pos, false);
    }
    // kv = kv_norm(wkv(x)); RoPE last 64; QAT-sim nope dims (group 64)
    let kv_pre = quant_gemm(x, s, dim, &w.wkv, hd, 128);
    let mut kv = rms_norm(&kv_pre, s, hd, &w.kv_norm, cfg.norm_eps);
    {
        let pos: Vec<usize> = (0..s).map(|i| start_pos + i).collect();
        apply_rope(&mut kv, s, hd, rope, &pos, false);
    }
    {
        let nope = hd - rd;
        // act_quant on the [.., :448] view: treat as rows of length 448
        let mut tmp = vec![0.0f32; s * nope];
        for i in 0..s {
            tmp[i * nope..(i + 1) * nope].copy_from_slice(&kv[i * hd..i * hd + nope]);
        }
        act_quant_sim(&mut tmp, s, nope, 64);
        for i in 0..s {
            kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
        }
    }
    (qr, q, kv)
}

/// Grouped-LoRA O (§B.1.4): de-rotation (caller does it), then
/// o.view(s,8,4096) einsum with wo_a.view(8,1024,4096) → wo_b.
pub fn attn_out_proj(w: &AttnWeights, o: &[f32], s: usize, cfg: &Dsv4Config) -> Vec<f32> {
    let g = cfg.o_groups;
    let r = cfg.o_lora_rank;
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let gd = nh * hd / g; // 4096 per group
    // einsum "bsgd,grd->bsgr": per group, [s, gd] @ wo_a[g]ᵀ -> [s, r] (bf16 GEMM)
    let mut oflat = vec![0.0f32; s * g * r];
    for grp in 0..g {
        let og = &o[grp * gd..]; // strided view per token — gather
        let mut xg = vec![0.0f32; s * gd];
        for i in 0..s {
            xg[i * gd..(i + 1) * gd].copy_from_slice(&og[i * nh * hd..i * nh * hd + gd]);
        }
        let wag = &w.wo_a[grp * r * gd..(grp + 1) * r * gd];
        let yg = gemm_bf16(&xg, s, gd, wag, r);
        for i in 0..s {
            oflat[i * g * r + grp * r..i * g * r + (grp + 1) * r].copy_from_slice(&yg[i * r..(i + 1) * r]);
        }
    }
    quant_gemm(&oflat, s, g * r, &w.wo_b, cfg.dim, 128)
}

/// Trunk attention forward (model.py:490-548). x [s, dim] = attn_norm output.
/// Returns (out [s, dim] bf16, topk_idxs flattened [s, T] with −1 masking).
#[allow(clippy::too_many_arguments)]
pub fn attn_forward(
    w: &AttnWeights,
    st: &mut AttnState,
    x: &[f32],
    s: usize,
    start_pos: usize,
    rope: &RopeTable,
    hadamard128: Option<&[f32]>,
    cfg: &Dsv4Config,
) -> (Vec<f32>, Vec<i64>, usize) {
    let win = cfg.window_size;
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let (qr, q, kv) = attn_qkv(w, x, s, start_pos, rope, cfg);
    // index lists
    let win_idxs = window_topk_idxs(win, s, start_pos);
    let topk: Vec<Vec<i64>> = match w.kind {
        LayerKind::Swa => win_idxs,
        LayerKind::Csa => {
            let offset = if start_pos == 0 { s } else { win };
            let iw = w.indexer.as_ref().unwrap();
            let ist = st.indexer.as_mut().unwrap();
            let comp = indexer_forward(iw, ist, x, s, &qr, start_pos, offset, rope, hadamard128.unwrap(), cfg);
            win_idxs
                .into_iter()
                .zip(comp)
                .map(|(mut a, b)| {
                    a.extend(b);
                    a
                })
                .collect()
        }
        LayerKind::Hca => {
            let offset = if start_pos == 0 { s } else { win };
            let comp = compress_topk_idxs(128, s, start_pos, offset);
            win_idxs
                .into_iter()
                .zip(comp)
                .map(|(mut a, b)| {
                    a.extend(b);
                    a
                })
                .collect()
        }
    };
    let t = topk[0].len();
    let scale = (hd as f64).powf(-0.5) as f32; // 512^-0.5
    let o: Vec<f32>;
    if start_pos == 0 {
        // ring write (rotated at s > win), then attention over current kv (+ compressor rows)
        if s <= win {
            st.kv_cache[..s * hd].copy_from_slice(&kv);
        } else {
            let cutoff = s % win;
            // cache[cutoff:win], cache[:cutoff] = kv[-win:].split([win-cutoff, cutoff])
            st.kv_cache[cutoff * hd..win * hd].copy_from_slice(&kv[(s - win) * hd..(s - cutoff) * hd]);
            st.kv_cache[..cutoff * hd].copy_from_slice(&kv[(s - cutoff) * hd..s * hd]);
        }
        let mut kv_attn = kv.clone();
        if let Some(comp) = st.compressor.as_mut() {
            let (head, tail) = st.kv_cache.split_at_mut(win * hd);
            let _ = head;
            if let Some(kvc) = comp.forward(x, s, 0, rope, cfg.norm_eps, tail) {
                kv_attn.extend_from_slice(&kvc);
            }
        }
        let flat: Vec<i64> = topk.iter().flatten().copied().collect();
        o = sparse_attn(&q, s, cfg.n_heads, hd, &kv_attn, kv_attn.len() / hd, &w.sink, &flat, t, scale);
    } else {
        // write-before-attention: current token attends to itself
        st.kv_cache[(start_pos % win) * hd..(start_pos % win + 1) * hd].copy_from_slice(&kv[..hd]);
        if let Some(comp) = st.compressor.as_mut() {
            let (head, tail) = st.kv_cache.split_at_mut(win * hd);
            let _ = head;
            comp.forward(x, s, start_pos, rope, cfg.norm_eps, tail);
        }
        let flat: Vec<i64> = topk.iter().flatten().copied().collect();
        o = sparse_attn(&q, s, cfg.n_heads, hd, &st.kv_cache, st.cache_rows, &w.sink, &flat, t, scale);
    }
    // de-rotation (inverse RoPE on the attention output, compensating K≡V leak)
    let mut o = o;
    {
        let rows = s * cfg.n_heads;
        let pos: Vec<usize> = (0..rows).map(|i| start_pos + i / cfg.n_heads).collect();
        apply_rope(&mut o, rows, hd, rope, &pos, true);
    }
    let _ = rd;
    let out = attn_out_proj(w, &o, s, cfg);
    let flat: Vec<i64> = topk.into_iter().flatten().collect();
    (out, flat, t)
}

// ---------------------------------------------------------------------------
// MoE (§B.9)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ExpertF32 {
    pub w1: Vec<f32>, // [inter, dim]
    pub w2: Vec<f32>, // [dim, inter]
    pub w3: Vec<f32>, // [inter, dim]
}

/// Routed-expert bank: NVFP4 tensors (Lane A repack) with lazy exact-f32 dequant
/// cache (256 slots, one per expert; get_or_init is thread-safe).
pub struct ExpertBank {
    pub w1: Vec<Nvfp4Tensor>,
    pub w2: Vec<Nvfp4Tensor>,
    pub w3: Vec<Nvfp4Tensor>,
    cache: Vec<OnceLock<ExpertF32>>,
}

impl ExpertBank {
    pub fn from_nvfp4(w1: Vec<Nvfp4Tensor>, w2: Vec<Nvfp4Tensor>, w3: Vec<Nvfp4Tensor>) -> Self {
        let n = w1.len();
        ExpertBank {
            w1,
            w2,
            w3,
            cache: (0..n).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Test path: pre-dequantized experts (no NVFP4 backing needed).
    pub fn from_f32(experts: Vec<ExpertF32>) -> Self {
        let n = experts.len();
        let cache: Vec<OnceLock<ExpertF32>> = (0..n).map(|_| OnceLock::new()).collect();
        for (slot, e) in cache.iter().zip(experts) {
            let _ = slot.set(e);
        }
        ExpertBank { w1: Vec::new(), w2: Vec::new(), w3: Vec::new(), cache }
    }

    pub fn get(&self, e: usize) -> &ExpertF32 {
        self.cache[e].get_or_init(|| {
            ExpertF32 {
                w1: crate::dsv4_load::dequant_nvfp4_f32(&self.w1[e]),
                w2: crate::dsv4_load::dequant_nvfp4_f32(&self.w2[e]),
                w3: crate::dsv4_load::dequant_nvfp4_f32(&self.w3[e]),
            }
        })
    }
}

pub struct MoeWeights {
    pub gate_w: Vec<f32>,            // [n_exp, dim] fp32 (bf16 upcast)
    pub gate_bias: Option<Vec<f32>>, // [n_exp] fp32 — selection only (layers ≥ 3)
    pub tid2eid: Option<Vec<i32>>,   // [vocab, 6] — hash layers 0–2
    pub shared: ExpertF32,           // fp8 dequant — inner block 128
    pub experts: ExpertBank,         // fp4 — inner block 32
}

/// `MoE.forward`: fp32 accumulator, per-expert row gather (expert-ascending
/// merge order, matching the reference's loop), shared expert added last,
/// output bf16. Returns (out [s,dim], router_w [s,6] fp32, router_idx [s,6]).
pub fn moe_forward(
    w: &MoeWeights,
    x: &[f32],
    s: usize,
    ids: &[i64],
    cfg: &Dsv4Config,
) -> (Vec<f32>, Vec<f32>, Vec<i64>) {
    let dim = cfg.dim;
    let inter = cfg.moe_inter_dim;
    let ne = cfg.n_routed_experts;
    let topk = cfg.n_activated_experts;
    let tid = w.tid2eid.as_deref().map(|t| (t, ids));
    let (weights, indices) = gate_forward(x, s, dim, &w.gate_w, w.gate_bias.as_deref(), tid, ne, topk, cfg.route_scale);
    // expert-major partials (parallel over experts; each expert's rows are disjoint)
    let mut partials: Vec<Vec<(usize, Vec<f32>)>> = (0..ne).map(|_| Vec::new()).collect();
    par_rows(&mut partials, |e, slot| {
        let mut rows: Vec<(usize, f32)> = Vec::new(); // (token, which slot)
        for i in 0..s {
            for j in 0..topk {
                if indices[i * topk + j] as usize == e {
                    rows.push((i, weights[i * topk + j]));
                }
            }
        }
        if rows.is_empty() {
            return;
        }
        let exp = w.experts.get(e);
        for (i, wgt) in rows {
            let xi = &x[i * dim..(i + 1) * dim];
            let y = expert_forward_token(xi, &exp.w1, &exp.w2, &exp.w3, dim, inter, 32, cfg.swiglu_limit, Some(wgt));
            slot.push((i, y));
        }
    });
    // fp32 accumulator; expert-ascending merge order (reference loop order)
    let mut y = vec![0.0f32; s * dim];
    for part in partials.iter() {
        for (i, row) in part {
            for d in 0..dim {
                y[i * dim + d] += row[d];
            }
        }
    }
    // shared expert on the original bf16 x, no routing weight
    for i in 0..s {
        let xi = &x[i * dim..(i + 1) * dim];
        let ys = expert_forward_token(xi, &w.shared.w1, &w.shared.w2, &w.shared.w3, dim, inter, 128, cfg.swiglu_limit, None);
        for d in 0..dim {
            y[i * dim + d] += ys[d];
        }
    }
    round_bf16(&mut y);
    (y, weights, indices)
}

