//! The block forward (B.8: hc_pre, norm, sublayer, hc_post, twice), the DSpark stages (B.10) and the
//! replay-driver piece outputs (written as .npy under the oracle's keys).
//! Siblings: `attn_moe.rs` (sublayers); `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------
// Block (§B.8 ordering: hc_pre -> norm -> sublayer -> hc_post, twice)
// ---------------------------------------------------------------------------

/// mHC head-collapse parameters (trunk `hc_head_*` and mtp.2's own).
#[derive(Debug, Clone)]
pub struct HcHeadParams {
    pub hc_fn: Vec<f32>,   // [4, hc*dim]
    pub hc_base: Vec<f32>, // [4]
    pub hc_scale: f32,     // [1]
}

pub struct CpuLayer {
    pub kind: LayerKind,
    pub attn: AttnWeights,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn: HcParams,
    pub hc_ffn: HcParams,
    pub moe: MoeWeights,
    // DSpark stage extras (None on trunk layers)
    pub main_proj: Option<Vec<f32>>, // [4096, 12288] fp8-dequant (stage 0)
    pub main_norm: Option<Vec<f32>>, // [4096] (stage 0)
    pub norm: Option<Vec<f32>>,      // [4096] (stage 2)
    pub hc_head: Option<HcHeadParams>, // (stage 2)
    pub markov_w1: Option<Vec<f32>>, // [vocab, 256] bf16-valued (stage 2)
    pub markov_w2: Option<Vec<f32>>, // [vocab, 256] fp32 (stage 2)
    pub confidence: Option<Vec<f32>>, // [4352] fp32 (stage 2)
}

#[derive(Debug, Default)]
pub struct BlockTrace {
    pub attn_out: Vec<f32>,   // [s, dim]
    pub ffn_out: Vec<f32>,    // [s, dim]
    pub router_w: Vec<f32>,   // [s, 6]
    pub router_idx: Vec<i64>, // [s, 6]
    pub topk_idx: Vec<i64>,   // [s, T] flattened
    pub topk_t: usize,
}

/// hc_pre over all tokens (parallel). x [s, hc*dim] flattened streams.
/// Returns (y [s, dim] bf16, posts [s], combs [s]).
pub fn hc_pre_all(x: &[f32], s: usize, p: &HcParams, cfg: &Dsv4Config) -> (Vec<f32>, Vec<[f32; 4]>, Vec<[f32; 16]>) {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let mut slots: Vec<(Vec<f32>, [f32; 4], [f32; 16])> = (0..s).map(|_| (Vec::new(), [0.0; 4], [0.0; 16])).collect();
    par_rows(&mut slots, |i, slot| {
        let xf = &x[i * hc * dim..(i + 1) * hc * dim];
        *slot = hc_pre_token(xf, hc, dim, p, cfg.norm_eps, cfg.hc_sinkhorn_iters, cfg.hc_eps);
    });
    let mut y = vec![0.0f32; s * dim];
    let mut posts = Vec::with_capacity(s);
    let mut combs = Vec::with_capacity(s);
    for (i, (yi, pi, ci)) in slots.into_iter().enumerate() {
        y[i * dim..(i + 1) * dim].copy_from_slice(&yi);
        posts.push(pi);
        combs.push(ci);
    }
    (y, posts, combs)
}

/// hc_post over all tokens: out [s, hc*dim] streams.
fn hc_post_all(x_out: &[f32], residual: &[f32], posts: &[[f32; 4]], combs: &[[f32; 16]], cfg: &Dsv4Config) -> Vec<f32> {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let s = posts.len();
    let mut out = vec![0.0f32; s * hc * dim];
    par_chunks(&mut out, hc * dim, |i, orow| {
        hc_post_token(
            &x_out[i * dim..(i + 1) * dim],
            &residual[i * hc * dim..(i + 1) * hc * dim],
            &posts[i],
            &combs[i],
            hc,
            dim,
            orow,
        );
    });
    out
}

/// Trunk `Block.forward`. x [s, hc*dim] streams (bf16-valued), ids [s].
/// Returns (new streams [s, hc*dim], trace).
#[allow(clippy::too_many_arguments)]
pub fn block_forward(
    layer: &CpuLayer,
    st: &mut AttnState,
    x: &[f32],
    s: usize,
    start_pos: usize,
    ids: &[i64],
    rope: &RopeTable,
    hadamard128: Option<&[f32]>,
    cfg: &Dsv4Config,
) -> (Vec<f32>, BlockTrace) {
    let dim = cfg.dim;
    let mut trace = BlockTrace::default();
    // --- attention sublayer ---
    let (y, posts, combs) = hc_pre_all(x, s, &layer.hc_attn, cfg);
    let yn = rms_norm(&y, s, dim, &layer.attn_norm, cfg.norm_eps);
    let (attn_out, topk, t) = attn_forward(&layer.attn, st, &yn, s, start_pos, rope, hadamard128, cfg);
    trace.attn_out = attn_out.clone();
    trace.topk_idx = topk;
    trace.topk_t = t;
    let x2 = hc_post_all(&attn_out, x, &posts, &combs, cfg);
    // --- ffn sublayer ---
    let (y2, posts2, combs2) = hc_pre_all(&x2, s, &layer.hc_ffn, cfg);
    let y2n = rms_norm(&y2, s, dim, &layer.ffn_norm, cfg.norm_eps);
    let (ffn_out, rw, ri) = moe_forward(&layer.moe, &y2n, s, ids, cfg);
    trace.ffn_out = ffn_out.clone();
    trace.router_w = rw;
    trace.router_idx = ri;
    let x3 = hc_post_all(&ffn_out, &x2, &posts2, &combs2, cfg);
    (x3, trace)
}

// ---------------------------------------------------------------------------
// DSpark (§B.10)
// ---------------------------------------------------------------------------

/// `DSparkAttention` prefill/warm branch: compute main_kv for ALL trunk
/// positions, RoPE 0..S−1, FP8-sim, ring-write. Returns nothing (h passes
/// through unchanged — no attention, no FFN in the warm pass).
pub fn dspark_attn_warm(
    w: &AttnWeights,
    kv_cache: &mut [f32], // [128, 512]
    main_x: &[f32],
    s: usize,
    rope: &RopeTable,
    cfg: &Dsv4Config,
) {
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let win = cfg.window_size;
    let mk = quant_gemm(main_x, s, cfg.dim, &w.wkv, hd, 128);
    let mut main_kv = rms_norm(&mk, s, hd, &w.kv_norm, cfg.norm_eps);
    let pos: Vec<usize> = (0..s).collect();
    apply_rope(&mut main_kv, s, hd, rope, &pos, false);
    let nope = hd - rd;
    let mut tmp = vec![0.0f32; s * nope];
    for i in 0..s {
        tmp[i * nope..(i + 1) * nope].copy_from_slice(&main_kv[i * hd..i * hd + nope]);
    }
    act_quant_sim(&mut tmp, s, nope, 64);
    for i in 0..s {
        main_kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
    }
    if s <= win {
        kv_cache[..s * hd].copy_from_slice(&main_kv);
    } else {
        let cutoff = s % win;
        kv_cache[cutoff * hd..win * hd].copy_from_slice(&main_kv[(s - win) * hd..(s - cutoff) * hd]);
        kv_cache[..cutoff * hd].copy_from_slice(&main_kv[(s - cutoff) * hd..s * hd]);
    }
}

/// `DSparkAttention` decode branch (model.py:771-792): main_kv at start_pos,
/// draft q/kv at positions start_pos+seqlen .. +block, 133-entry non-causal
/// block index list, sparse_attn, draft-freq de-rotation, grouped-O.
/// x [block, dim] is the collapsed+normed draft input. main_x [1, dim].
#[allow(clippy::too_many_arguments)]
pub fn dspark_attn_forward(
    w: &AttnWeights,
    kv_cache: &mut [f32], // [128, 512]
    x: &[f32],
    block: usize,
    start_pos: usize,
    main_x: &[f32],
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let win = cfg.window_size;
    let sm = main_x.len() / cfg.dim;
    // main kv at start_pos (sm = 1 for the draft pass)
    dspark_attn_warm_deque(w, kv_cache, main_x, sm, start_pos, rope, cfg);
    // draft q/kv at positions start_pos+sm .. start_pos+sm+block-1
    let pos0 = start_pos + sm;
    let (_qr, q, kv) = attn_qkv(w, x, block, pos0, rope, cfg);
    // index list: cat([arange(min(128, start_pos+1)), 128+arange(block)]) × block rows
    let idx_row = dspark_topk_idxs(win, block, start_pos);
    let t = idx_row.len();
    let mut flat = Vec::with_capacity(block * t);
    for _ in 0..block {
        flat.extend_from_slice(&idx_row);
    }
    let mut kv_cat = kv_cache.to_vec();
    kv_cat.extend_from_slice(&kv);
    let scale = (hd as f64).powf(-0.5) as f32;
    let mut o = sparse_attn(&q, block, cfg.n_heads, hd, &kv_cat, win + block, &w.sink, &flat, t, scale);
    // de-rotation with the DRAFT freqs
    let rows = block * cfg.n_heads;
    let pos: Vec<usize> = (0..rows).map(|i| pos0 + i / cfg.n_heads).collect();
    apply_rope(&mut o, rows, hd, rope, &pos, true);
    let _ = rd;
    attn_out_proj(w, &o, block, cfg)
}

/// main_kv single-position write used by the decode branch (RoPE at start_pos,
/// FP8-sim, ring slot start_pos % 128 — written BEFORE the draft attention).
fn dspark_attn_warm_deque(
    w: &AttnWeights,
    kv_cache: &mut [f32],
    main_x: &[f32],
    sm: usize,
    start_pos: usize,
    rope: &RopeTable,
    cfg: &Dsv4Config,
) {
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let mk = quant_gemm(main_x, sm, cfg.dim, &w.wkv, hd, 128);
    let mut main_kv = rms_norm(&mk, sm, hd, &w.kv_norm, cfg.norm_eps);
    let pos: Vec<usize> = (0..sm).map(|i| start_pos + i).collect();
    apply_rope(&mut main_kv, sm, hd, rope, &pos, false);
    let nope = hd - rd;
    let mut tmp = vec![0.0f32; sm * nope];
    for i in 0..sm {
        tmp[i * nope..(i + 1) * nope].copy_from_slice(&main_kv[i * hd..i * hd + nope]);
    }
    act_quant_sim(&mut tmp, sm, nope, 64);
    for i in 0..sm {
        main_kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
    }
    kv_cache[(start_pos % cfg.window_size) * hd..(start_pos % cfg.window_size + 1) * hd]
        .copy_from_slice(&main_kv[..hd]);
}

/// `DSparkBlock.forward`: warm (start_pos == 0) writes the ring and returns h
/// unchanged; decode runs the full Block (hc_pre → norm → DSparkAttention →
/// hc_post → hc_pre → norm → MoE → hc_post).
#[allow(clippy::too_many_arguments)]
pub fn dspark_block_forward(
    layer: &CpuLayer,
    kv_cache: &mut [f32],
    h: &[f32], // [block, hc*dim] streams
    block: usize,
    start_pos: usize,
    main_x: &[f32],
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> Vec<f32> {
    dspark_block_forward_traced(layer, kv_cache, h, block, start_pos, main_x, rope, cfg).0
}

/// Traced variant (A/B debugging): also returns the sublayer outputs.
#[allow(clippy::too_many_arguments)]
pub fn dspark_block_forward_traced(
    layer: &CpuLayer,
    kv_cache: &mut [f32],
    h: &[f32], // [block, hc*dim] streams
    block: usize,
    start_pos: usize,
    main_x: &[f32],
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> (Vec<f32>, BlockTrace) {
    let dim = cfg.dim;
    let mut trace = BlockTrace::default();
    if start_pos == 0 {
        let s = main_x.len() / dim;
        dspark_attn_warm(&layer.attn, kv_cache, main_x, s, rope, cfg);
        return (h.to_vec(), trace);
    }
    let (y, posts, combs) = hc_pre_all(h, block, &layer.hc_attn, cfg);
    let yn = rms_norm(&y, block, dim, &layer.attn_norm, cfg.norm_eps);
    let attn_out = dspark_attn_forward(&layer.attn, kv_cache, &yn, block, start_pos, main_x, rope, cfg);
    trace.attn_out = attn_out.clone();
    let h2 = hc_post_all(&attn_out, h, &posts, &combs, cfg);
    let (y2, posts2, combs2) = hc_pre_all(&h2, block, &layer.hc_ffn, cfg);
    let y2n = rms_norm(&y2, block, dim, &layer.ffn_norm, cfg.norm_eps);
    // stages are never hash-routed; ids unused (bias path)
    let ids = vec![0i64; block];
    let (ffn_out, rw, ri) = moe_forward(&layer.moe, &y2n, block, &ids, cfg);
    trace.ffn_out = ffn_out.clone();
    trace.router_w = rw;
    trace.router_idx = ri;
    (hc_post_all(&ffn_out, &h2, &posts2, &combs2, cfg), trace)
}

/// argmax, first-max-index (matches torch.argmax tie behavior claim; f32 logits
/// ties are measure-zero). Used for the temperature=0 oracle sampler.
pub fn argmax_first(x: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as i64
}

// ---------------------------------------------------------------------------
// Piece outputs (the replay driver writes these as .npy under the same keys)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct PieceOutputs {
    pub f32_arrays: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub i64_arrays: Vec<(String, Vec<usize>, Vec<i64>)>,
}

impl PieceOutputs {
    pub fn push_f32(&mut self, key: &str, shape: &[usize], data: Vec<f32>) {
        self.f32_arrays.push((key.to_string(), shape.to_vec(), data));
    }
    pub fn push_i64(&mut self, key: &str, shape: &[usize], data: Vec<i64>) {
        self.i64_arrays.push((key.to_string(), shape.to_vec(), data));
    }
}

/// Layer piece (swa/csa/hca): prefill over pre.x [S, hc*dim] + D decode steps
/// over dec_xs[i] [1, hc*dim]; ids [S+D]. Emits keys exactly like the oracle
/// npz (pre.*, dec{i}.*, kv_cache). Decode inputs are independent synthetic
/// per step (the replay contract).
pub fn run_layer_piece(
    cfg: &Dsv4Config,
    layer: &CpuLayer,
    ids: &[i64],
    pre_x: &[f32],
    dec_xs: &[Vec<f32>],
    max_seq_len: usize,
) -> PieceOutputs {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let s = pre_x.len() / (hc * dim);
    let d = dec_xs.len();
    assert_eq!(ids.len(), s + d);
    // RoPE table for this layer kind; positions cover prefill + decode + 1
    let positions = s + d + 8;
    let rope = layer_rope_table(cfg, layer.kind, positions);
    let had = if matches!(layer.kind, LayerKind::Csa) {
        Some(hadamard_scaled(cfg.index_head_dim))
    } else {
        None
    };
    let mut st = AttnState::new(cfg, &layer.attn, max_seq_len);
    let mut out = PieceOutputs::default();
    // prefill
    let (y, tr) = block_forward(layer, &mut st, pre_x, s, 0, &ids[..s], &rope, had.as_deref(), cfg);
    out.push_f32("pre.y", &[s, hc, dim], y);
    out.push_f32("pre.attn_out", &[s, dim], tr.attn_out);
    out.push_f32("pre.ffn_out", &[s, dim], tr.ffn_out);
    out.push_f32("pre.router_w", &[s, cfg.n_activated_experts], tr.router_w);
    out.push_i64("pre.router_idx", &[s, cfg.n_activated_experts], tr.router_idx);
    out.push_i64("pre.topk_idx", &[1, s, tr.topk_t], tr.topk_idx);
    // decode steps
    for (i, dx) in dec_xs.iter().enumerate() {
        assert_eq!(dx.len(), hc * dim);
        let sp = s + i;
        let (y, tr) = block_forward(layer, &mut st, dx, 1, sp, &ids[sp..sp + 1], &rope, had.as_deref(), cfg);
        out.push_f32(&format!("dec{i}.y"), &[1, hc, dim], y);
        out.push_f32(&format!("dec{i}.attn_out"), &[1, dim], tr.attn_out);
        // oracle saves dec router_w squeezed to [6], router_idx as [1,6]
        out.push_f32(&format!("dec{i}.router_w"), &[cfg.n_activated_experts], tr.router_w);
        out.push_i64(&format!("dec{i}.router_idx"), &[1, cfg.n_activated_experts], tr.router_idx);
        out.push_i64(&format!("dec{i}.topk_idx"), &[1, 1, tr.topk_t], tr.topk_idx);
    }
    out.push_f32("kv_cache", &[st.cache_rows, cfg.head_dim], st.kv_cache);
    out
}

/// RoPE table per layer kind (§B.1.3): SWA plain θ=10000 (YaRN force-disabled);
/// CSA/HCA YaRN θ=160000, original_seq_len=65536, factor 16, β 32/1.
pub fn layer_rope_table(cfg: &Dsv4Config, kind: LayerKind, positions: usize) -> RopeTable {
    match kind {
        LayerKind::Swa => rope_table(cfg.rope_head_dim, positions, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
        _ => rope_table(
            cfg.rope_head_dim,
            positions,
            cfg.original_seq_len,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        ),
    }
}

/// DSpark piece: warm all 3 stages' rings from warm_main_hidden [Sw, 3*dim],
/// then the 5-position draft at start_pos = Sw. Emits draft.h_in, draft.h0..2,
/// draft.output_ids, draft.logits, draft.confidence.
pub fn run_dspark_piece(
    cfg: &Dsv4Config,
    stages: &[CpuLayer; 3],
    embed: &[f32], // [vocab, dim] bf16-valued
    head: &[f32],  // [vocab, dim] fp32
    warm_main_hidden: &[f32],
    draft_main_hidden: &[f32],
    real_token: i64,
) -> PieceOutputs {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let block = cfg.dspark_block_size;
    let sw = warm_main_hidden.len() / (3 * dim);
    let positions = sw + block + 8;
    let rope = layer_rope_table(cfg, LayerKind::Swa, positions);
    let mut rings: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0f32; cfg.window_size * cfg.head_dim]).collect();
    // warm: main_x for ALL trunk positions; all 3 stages ring-write
    let st0 = &stages[0];
    let main_proj = st0.main_proj.as_ref().unwrap();
    let main_norm = st0.main_norm.as_ref().unwrap();
    let mxw = quant_gemm(warm_main_hidden, sw, 3 * dim, main_proj, dim, 128);
    let main_x_w = rms_norm(&mxw, sw, dim, main_norm, cfg.norm_eps);
    let mut out = PieceOutputs::default();
    {
        // h is computed by forward_embed but unused by the warm pass
        let draft_ids: Vec<i64> = std::iter::once(real_token)
            .chain(std::iter::repeat(cfg.dspark_noise_token_id as i64).take(block - 1))
            .collect();
        let h = dspark_embed(embed, &draft_ids, hc, dim);
        for (i, st) in stages.iter().enumerate() {
            let _ = dspark_block_forward(st, &mut rings[i], &h, block, 0, &main_x_w, &rope, cfg);
        }
    }
    // draft pass at start_pos = sw
    let mxd = quant_gemm(draft_main_hidden, 1, 3 * dim, main_proj, dim, 128);
    let main_x_d = rms_norm(&mxd, 1, dim, main_norm, cfg.norm_eps);
    let draft_ids: Vec<i64> = std::iter::once(real_token)
        .chain(std::iter::repeat(cfg.dspark_noise_token_id as i64).take(block - 1))
        .collect();
    let mut h = dspark_embed(embed, &draft_ids, hc, dim);
    out.push_f32("draft.h_in", &[block, hc, dim], h.clone());
    for (i, st) in stages.iter().enumerate() {
        h = dspark_block_forward(st, &mut rings[i], &h, block, sw, &main_x_d, &rope, cfg);
        out.push_f32(&format!("draft.h{i}"), &[block, hc, dim], h.clone());
    }
    // forward_head (mtp.2): hc_head -> norm -> LM head (fp32) -> Markov chain -> confidence
    let st2 = &stages[2];
    let hch = st2.hc_head.as_ref().unwrap();
    let mut collapse = vec![0.0f32; block * dim];
    for i in 0..block {
        let xf = &h[i * hc * dim..(i + 1) * hc * dim];
        let y = hc_head_token(xf, hc, dim, &hch.hc_fn, &hch.hc_base, hch.hc_scale, cfg.norm_eps, cfg.hc_eps);
        collapse[i * dim..(i + 1) * dim].copy_from_slice(&y);
    }
    let yn = rms_norm(&collapse, block, dim, st2.norm.as_ref().unwrap(), cfg.norm_eps);
    let vocab = cfg.vocab_size;
    let mut logits = gemm_f32(&yn, block, dim, head, vocab); // fp32 [block, vocab]
    // sequential Markov bigram chain over output_ids [6]
    let mw1 = st2.markov_w1.as_ref().unwrap();
    let mw2 = st2.markov_w2.as_ref().unwrap();
    let rank = cfg.dspark_markov_rank;
    let mut output_ids = vec![0i64; block + 1];
    output_ids[0] = real_token;
    let mut markov_embeds: Vec<Vec<f32>> = Vec::with_capacity(block);
    for i in 0..block {
        let id = output_ids[i] as usize;
        let e: Vec<f32> = mw1[id * rank..(id + 1) * rank].to_vec(); // bf16-valued embedding
        let bias = gemm_f32(&e, 1, rank, mw2, vocab); // fp32 bigram bias
        for v in 0..vocab {
            logits[i * vocab + v] += bias[v];
        }
        markov_embeds.push(e);
        output_ids[i + 1] = argmax_first(&logits[i * vocab..(i + 1) * vocab]);
    }
    // confidence head: cat([collapse, markov_embeds], -1) -> fp32 Linear, raw score
    let conf_w = st2.confidence.as_ref().unwrap();
    let mut confidence = vec![0.0f32; block];
    for i in 0..block {
        let mut catv = Vec::with_capacity(dim + rank);
        catv.extend_from_slice(&collapse[i * dim..(i + 1) * dim]);
        catv.extend_from_slice(&markov_embeds[i]);
        confidence[i] = dot_tree(&catv, conf_w);
    }
    out.push_i64("draft.output_ids", &[block + 1], output_ids);
    out.push_f32("draft.logits", &[block, vocab], logits);
    out.push_f32("draft.confidence", &[block], confidence);
    out
}

/// Embed draft ids and replicate to hc streams (forward_embed).
fn dspark_embed(embed: &[f32], ids: &[i64], hc: usize, dim: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; ids.len() * hc * dim];
    for (i, &id) in ids.iter().enumerate() {
        let row = &embed[id as usize * dim..(id as usize + 1) * dim];
        for s in 0..hc {
            h[(i * hc + s) * dim..(i * hc + s + 1) * dim].copy_from_slice(row);
        }
    }
    round_bf16(&mut h); // embedding output is bf16
    h
}

/// Head piece (trunk collapse): hc_head (sigmoid-only) -> RMSNorm -> fp32 LM
/// head on the LAST row. x [S, hc*dim]. Emits collapsed [S, dim], logits [vocab].
pub fn run_head_piece(
    cfg: &Dsv4Config,
    hc_head: &HcHeadParams,
    norm: &[f32],
    head: &[f32],
    x: &[f32],
) -> PieceOutputs {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let s = x.len() / (hc * dim);
    let mut collapse = vec![0.0f32; s * dim];
    for i in 0..s {
        let xf = &x[i * hc * dim..(i + 1) * hc * dim];
        let y = hc_head_token(xf, hc, dim, &hc_head.hc_fn, &hc_head.hc_base, hc_head.hc_scale, cfg.norm_eps, cfg.hc_eps);
        collapse[i * dim..(i + 1) * dim].copy_from_slice(&y);
    }
    let yn = rms_norm(&collapse, s, dim, norm, cfg.norm_eps);
    let logits = gemm_f32(&yn[(s - 1) * dim..s * dim], 1, dim, head, cfg.vocab_size);
    let mut out = PieceOutputs::default();
    out.push_f32("collapsed", &[s, dim], collapse);
    out.push_f32("logits", &[cfg.vocab_size], logits);
    out
}

