//! Dsv4DSpark — the DSpark drafter (Phase 5, §B.10). Three full SWA-MoE stages fed by
//! `main_norm(main_proj(h.mean(streams) @ layers 40/41/42))`.
//!
//! **Single-process on the HEAD only** (full 256 experts; the verify rides the TP=2 trunk).
//! The draft's numerics are independent of the trunk's TP sharding — running it unsharded on
//! rank 0 is correct and avoids loading the DSpark stages on the node + a draft all-reduce.
//!
//! Lifecycle: `warm(main_hidden_prefill)` once after trunk prefill → `draft(main_hidden_decode,
//! real_token, start_pos)` each decode step → returns `block` draft tokens (+ raw confidence,
//! logged but unused for v1 gating). Mechanics validated vs `dsv4_cpu::run_dspark_piece`
//! (shapes / warm path / 133-entry index list / Markov chain structure — NOT value equality:
//! the §7 G1 amendment governs, the draft chain is intrinsically chaotic).

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

use crate::dsv4_attn::{Dsv4AttnRuntime, Dsv4AttnState, Dsv4GpuLayer, Fp8Weight, B, S};
use crate::dsv4_gpu::{env_flag_once, GB, GS};
use crate::dsv4_graph::{DecodeGraphs, Slot, Variant};
use crate::dsv4_launch;
use crate::dsv4_load::{self, Dsv4Config, HostTensor};
use crate::gpu::MoeGroupedScratch;
use crate::quant;

/// The DSpark drafter (3 stages + stage extras). Owns its own single-process runtime.
pub struct Dsv4DSpark {
    pub dev: Arc<CudaDevice>,
    pub cfg: Dsv4Config,
    pub rt: Dsv4AttnRuntime,
    pub stages: Vec<Dsv4GpuLayer>,
    pub states: Vec<Dsv4AttnState>,
    pub scratch: MoeGroupedScratch,
    /// Shared trunk embed/head (cloned Arc handles from the trunk model — cheap).
    pub embed: B,
    pub head: B,
    // stage-0 extras
    main_proj: Fp8Weight, // [dim, 3*dim] FP8 (raw codes, MMA-repacked)
    main_norm: S,         // [dim] f32
    // stage-2 extras
    stage2_norm: S,       // [dim] f32
    hc_head_fn: S,        // [hc, hc*dim] f32
    hc_head_base: S,      // [hc] f32
    hc_head_scale: S,     // [1] f32
    markov_w1: B,         // [vocab, rank] bf16 (embedding)
    markov_w2: B,         // [vocab, rank] bf16 (bf16-exact f32 values; gemm_binv_f32_b)
    confidence: S,        // [dim+rank] f32
    /// Host cache of markov_w1 for the sequential Markov gather (~66 MB; cheap lookups).
    markov_w1_host: Vec<bf16>,
    block: usize,
    noise_id: i32,
    // ---- CUDA-graph draft (--dspark-graph=1): persistent inputs + the draft graph ----
    /// Persistent zeros [block] for the MoE router's unused ids arg (tid2eid is None here).
    g_zeros: Option<CudaSlice<i32>>,
    /// Persistent main_hidden [1, 3*dim] (d2d from the trunk's per-step capture).
    g_main_hidden: Option<B>,
    /// Persistent position buffers: main_kv sp [1], draft q positions [block*nh], draft kv
    /// positions [block] — refreshed per step via dsv4_iota_b launches OUTSIDE the graph.
    g_pos_sp: Option<CudaSlice<i32>>,
    g_pos_q: Option<CudaSlice<i32>>,
    g_pos_kv: Option<CudaSlice<i32>>,
    /// Persistent collapse readout [block, dim] (the graph's second memcpy node writes it;
    /// the confidence head consumes it on the host).
    g_collapse_out: Option<B>,
    dgraphs: Option<DecodeGraphs>,
    /// --dspark-phase-ms accumulators: GPU-synced chain time vs host Markov-tail time.
    pub t_chain: f64,
    pub t_markov: f64,
    /// --dspark-fp8-logits: fp8_bsb copies of the draft LM head + Markov W2 (halve the
    /// draft's head reads — the vLLM stack's FP8 DeepGEMM copies, memo §2). Draft-side only:
    /// near-tie argmax flips cost acceptance, never correctness (the trunk verify governs).
    head_fp8: Option<Fp8Weight>,
    markov_w2_fp8: Option<Fp8Weight>,
    /// Per-call arm select (defaults to the env flag; tests flip it for A/B). MUST NOT
    /// change after the draft graph is captured — the graph bakes one arm's output copy
    /// (guarded below).
    pub use_fp8_logits: bool,
    g_graph_fp8_arm: bool,
    /// Persistent bf16 logits readout [block*vocab] for the graphed fp8 arm (the fp8 GEMM
    /// outputs bf16; the f32 eager buffer is used on the non-fp8 arm).
    g_logits_bf16: Option<B>,
}

pub struct DraftOut {
    /// The `block` drafted tokens (output_ids[1..=block]).
    pub drafts: Vec<i32>,
    /// Raw confidence score per draft position (fp32, no sigmoid; logged, unused v1).
    pub confidence: Vec<f32>,
}

/// The graphed draft chain's logits, pre-readout: fp32 (bf16 head path) or bf16 (the
/// --dspark-fp8-logits fp8_bsb path — the fp8 GEMM's native output).
pub enum DraftLogitsDev {
    F32(GS),
    BF16(GB),
}

impl Dsv4DSpark {
    /// Load the 3 DSpark stages from the bundle + the stage extras. `embed`/`head` are cloned
    /// from the trunk model (tied references). Single-process (world=1) — full 256 experts.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dev: &Arc<CudaDevice>,
        bundle: &Path,
        cfg: &Dsv4Config,
        max_seq_len: usize,
        embed: B,
        head: B,
    ) -> Result<Self> {
        let positions = max_seq_len.max(64) + 16;
        let rt = Dsv4AttnRuntime::new_multikind(dev, positions, cfg)
            .context("Dsv4AttnRuntime::new_multikind (dspark)")?;

        let mut stages = Vec::with_capacity(cfg.n_mtp_layers);
        let mut states = Vec::with_capacity(cfg.n_mtp_layers);
        let mut main_proj = None;
        let mut main_norm_w = None;
        let mut stage2_norm_w = None;
        let mut hc_head_fn = None;
        let mut hc_head_base = None;
        let mut hc_head_scale = None;
        let mut markov_w1_dev = None;
        let mut markov_w2_dev = None;
        let mut confidence_w = None;
        let mut markov_w1_host = Vec::new();

        for stage in 0..cfg.n_mtp_layers {
            eprintln!("[dspark] loading stage {stage} (mtp.{stage}) ...");
            let (gl, mut extras) = rt
                .upload_mtp_stage(bundle, cfg, stage)
                .with_context(|| format!("upload_mtp_stage {stage}"))?;
            let st = rt.new_state_swa(cfg, 320)?;
            if stage == 0 {
                // main_proj: read raw FP8 from the bundle (the strict load dequants it to f32;
                // we need the codes for fp8_bsb). [dim, 3*dim] = [4096, 12288].
                let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, "mtp.0.main_proj.weight")
                    .context("read_raw_fp8 mtp.0.main_proj.weight")?;
                let (mp_m, mp_k) = (shape[0], shape[1]);
                let mp_wt = quant::repack_fp8_mma(&codes, mp_m, mp_k);
                main_proj = Some(Fp8Weight {
                    wt: dev.htod_sync_copy(&mp_wt)?,
                    sb: dev.htod_sync_copy(&sb)?,
                    m: mp_m, k: mp_k,
                });
                main_norm_w = Some(take_f32_extra(dev, &mut extras, "main_norm.weight", cfg.dim)?);
            }
            if stage == cfg.n_mtp_layers - 1 {
                let rank = cfg.dspark_markov_rank;
                stage2_norm_w = Some(take_f32_extra(dev, &mut extras, "norm.weight", cfg.dim)?);
                hc_head_fn = Some(take_f32_extra(dev, &mut extras, "hc_head_fn", cfg.hc_mult * cfg.hc_mult * cfg.dim)?);
                hc_head_base = Some(take_f32_extra(dev, &mut extras, "hc_head_base", cfg.hc_mult)?);
                hc_head_scale = Some(take_f32_extra(dev, &mut extras, "hc_head_scale", 1)?);
                // markov_w1: bf16 embedding [vocab, rank] (kept bf16 by the cast rule).
                let mw1 = take_bf16_extra(&mut extras, "markov_head.markov_w1.weight", cfg.vocab_size * rank)?;
                markov_w1_host = mw1.clone();
                markov_w1_dev = Some(dev.htod_sync_copy(&mw1)?);
                // markov_w2: bf16-exact f32 → downcast to bf16 for gemm_binv_f32_b (§12.A.4).
                let mw2_f32 = take_f32_data(&mut extras, "markov_head.markov_w2.weight", cfg.vocab_size * rank)?;
                let mw2_bf16: Vec<bf16> = mw2_f32.iter().map(|&v| bf16::from_f32(v)).collect();
                markov_w2_dev = Some(dev.htod_sync_copy(&mw2_bf16)?);
                confidence_w = Some(take_f32_extra(dev, &mut extras, "confidence_head.proj.weight", cfg.dim + rank)?);
            }
            stages.push(gl);
            states.push(st);
            eprintln!("[dspark]   stage {stage} loaded");
        }

        let mut me = Self {
            dev: dev.clone(),
            cfg: cfg.clone(),
            rt,
            stages,
            states,
            scratch: crate::gpu::new_moe_grouped_scratch_raw(dev, cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.n_activated_experts, 64, 16 * cfg.n_activated_experts),
            embed,
            head,
            main_proj: main_proj.expect("stage 0 main_proj"),
            main_norm: main_norm_w.expect("stage 0 main_norm"),
            stage2_norm: stage2_norm_w.expect("stage 2 norm"),
            hc_head_fn: hc_head_fn.expect("stage 2 hc_head_fn"),
            hc_head_base: hc_head_base.expect("stage 2 hc_head_base"),
            hc_head_scale: hc_head_scale.expect("stage 2 hc_head_scale"),
            markov_w1: markov_w1_dev.expect("stage 2 markov_w1"),
            markov_w2: markov_w2_dev.expect("stage 2 markov_w2"),
            confidence: confidence_w.expect("stage 2 confidence"),
            markov_w1_host,
            block: cfg.dspark_block_size,
            noise_id: cfg.dspark_noise_token_id as i32,
            g_zeros: None,
            g_main_hidden: None,
            g_pos_sp: None,
            g_pos_q: None,
            g_pos_kv: None,
            g_collapse_out: None,
            dgraphs: None,
            t_chain: 0.0,
            t_markov: 0.0,
            head_fp8: None,
            markov_w2_fp8: None,
            use_fp8_logits: false,
            g_graph_fp8_arm: false,
            g_logits_bf16: None,
        };
        me.maybe_make_fp8_heads()?;
        Ok(me)
    }

    /// Load from a converted `dspark.safetensors` artifact (the TP=2 path — both ranks load the
    /// SAME replicated artifact from their `{rank_dir}/dspark.safetensors`). Byte-identical to
    /// [`load`](Self::load) (the converter writes the exact bytes `upload_mtp_stage` uploads).
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_artifact(
        dev: &Arc<CudaDevice>,
        rank_dir: &Path,
        cfg: &Dsv4Config,
        max_seq_len: usize,
        embed: B,
        head: B,
    ) -> Result<Self> {
        use crate::dsv4_convert::ArtFile;
        use crate::gpu::Dsv4MoeGpu;
        let positions = max_seq_len.max(64) + 16;
        let rt = Dsv4AttnRuntime::new_multikind(dev, positions, cfg)
            .context("Dsv4AttnRuntime::new_multikind (dspark artifact)")?;
        let mut stages = Vec::with_capacity(cfg.n_mtp_layers);
        let mut states = Vec::with_capacity(cfg.n_mtp_layers);
        let mut main_proj_v = None;
        let mut main_norm_v = None;
        let mut stage2_norm_v = None;
        let mut hc_head_fn_v = None;
        let mut hc_head_base_v = None;
        let mut hc_head_scale_v = None;
        let mut markov_w1_dev = None;
        let mut markov_w2_dev = None;
        let mut confidence_v = None;
        let mut markov_w1_host = Vec::new();
        let up = |v: Vec<f32>| -> Result<S> { Ok(dev.htod_sync_copy(&v)?) };
        for stage in 0..cfg.n_mtp_layers {
            eprintln!("[dspark] loading stage {stage} from artifact ...");
            // Open ONE stage file at a time — ArtFile::open reads the whole file into RAM; opening
            // per-stage (not a combined 11.6 GB file) caps the host peak at ~3.8 GB and drops it
            // before the next stage (critical under the 84 GB trunk on unified memory).
            let art = ArtFile::open(&rank_dir.join(format!("dspark_stage{stage}.safetensors")))
                .with_context(|| format!("ArtFile::open {}/dspark_stage{stage}.safetensors", rank_dir.display()))?;
            let fp8w = |name: &str| -> Result<Fp8Weight> {
                let wt_name = format!("{name}.wt");
                let (m, k) = art.mk_of(&wt_name)?;
                Ok(Fp8Weight {
                    wt: dev.htod_sync_copy(art.u8_slice(&wt_name)?)?,
                    sb: dev.htod_sync_copy(art.u8_slice(&format!("{name}.sb"))?)?,
                    m, k,
                })
            };
            let f32_of = |name: &str| -> Result<Vec<f32>> { art.f32_of(name) };
            let (ne, hdim, inter) = (cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim);
            let moe = Dsv4MoeGpu {
                gu_wt: dev.htod_sync_copy(art.u8_slice("moe.gu_wt")?)?,
                gu_st: dev.htod_sync_copy(art.u8_slice("moe.gu_st")?)?,
                gu_gs: dev.htod_sync_copy(&art.f32_of("moe.gu_gs")?)?,
                dn_wt: dev.htod_sync_copy(art.u8_slice("moe.dn_wt")?)?,
                dn_st: dev.htod_sync_copy(art.u8_slice("moe.dn_st")?)?,
                dn_gs: dev.htod_sync_copy(&art.f32_of("moe.dn_gs")?)?,
                ne, h: hdim, inter, e_base: 0, e_span: ne,
            };
            let wo_a_bf = art.bf16_of("wo_a")?;
            let wo_a_q = {
                let (wt, sb) = crate::quant::quantize_fp8_bsb(&wo_a_bf, cfg.o_groups * cfg.o_lora_rank, cfg.dim);
                Fp8Weight {
                    wt: dev.htod_sync_copy(&wt)?,
                    sb: dev.htod_sync_copy(&sb)?,
                    m: cfg.o_groups * cfg.o_lora_rank,
                    k: cfg.dim,
                }
            };
            let gl = Dsv4GpuLayer {
                kind: dsv4_load::LayerKind::Swa,
                wq_a: fp8w("wq_a")?, wq_b: fp8w("wq_b")?, wkv: fp8w("wkv")?, wo_b: fp8w("wo_b")?,
                sh_gu: fp8w("sh_gu")?, sh_w2: fp8w("sh_w2")?,
                wo_a: dev.htod_sync_copy(&wo_a_bf)?,
                wo_a_q: Some(wo_a_q),
                q_norm: up(f32_of("q_norm")?)?, kv_norm: up(f32_of("kv_norm")?)?,
                attn_norm: up(f32_of("attn_norm")?)?, ffn_norm: up(f32_of("ffn_norm")?)?,
                sink: up(f32_of("attn_sink")?)?,
                hc_attn_fn: up(f32_of("hc_attn_fn")?)?, hc_attn_base: up(f32_of("hc_attn_base")?)?,
                hc_attn_scale: up(f32_of("hc_attn_scale")?)?,
                hc_ffn_fn: up(f32_of("hc_ffn_fn")?)?, hc_ffn_base: up(f32_of("hc_ffn_base")?)?,
                hc_ffn_scale: up(f32_of("hc_ffn_scale")?)?,
                gate_w: up(f32_of("gate_w")?)?,
                tid2eid: None,
                gate_bias: Some(up(f32_of("gate_bias")?)?),
                moe, comp_load: None, idx_load: None,
            };
            let st = rt.new_state_swa(cfg, 320)?;
            if stage == 0 {
                main_proj_v = Some(fp8w("main_proj")?);
                main_norm_v = Some(up(f32_of("main_norm")?)?);
            }
            if stage == cfg.n_mtp_layers - 1 {
                let rank = cfg.dspark_markov_rank;
                stage2_norm_v = Some(up(f32_of("norm")?)?);
                hc_head_fn_v = Some(up(f32_of("hc_head_fn")?)?);
                hc_head_base_v = Some(up(f32_of("hc_head_base")?)?);
                hc_head_scale_v = Some(up(f32_of("hc_head_scale")?)?);
                let mw1 = art.bf16_of("markov_w1")?;
                anyhow::ensure!(mw1.len() == cfg.vocab_size * rank, "markov_w1 len");
                markov_w1_host = mw1.clone();
                markov_w1_dev = Some(dev.htod_sync_copy(&mw1)?);
                let mw2_f32 = f32_of("markov_w2")?;
                let mw2_bf16: Vec<bf16> = mw2_f32.iter().map(|&v| bf16::from_f32(v)).collect();
                markov_w2_dev = Some(dev.htod_sync_copy(&mw2_bf16)?);
                confidence_v = Some(up(f32_of("confidence")?)?);
            }
            stages.push(gl);
            states.push(st);
            eprintln!("[dspark]   stage {stage} loaded (artifact)");
        }
        let mut me = Self {
            dev: dev.clone(), cfg: cfg.clone(), rt, stages, states,
            scratch: crate::gpu::new_moe_grouped_scratch_raw(dev, cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.n_activated_experts, 64, 16 * cfg.n_activated_experts),
            embed, head,
            main_proj: main_proj_v.expect("stage 0 main_proj"),
            main_norm: main_norm_v.expect("stage 0 main_norm"),
            stage2_norm: stage2_norm_v.expect("stage 2 norm"),
            hc_head_fn: hc_head_fn_v.expect("stage 2 hc_head_fn"),
            hc_head_base: hc_head_base_v.expect("stage 2 hc_head_base"),
            hc_head_scale: hc_head_scale_v.expect("stage 2 hc_head_scale"),
            markov_w1: markov_w1_dev.expect("stage 2 markov_w1"),
            markov_w2: markov_w2_dev.expect("stage 2 markov_w2"),
            confidence: confidence_v.expect("stage 2 confidence"),
            markov_w1_host,
            block: cfg.dspark_block_size,
            noise_id: cfg.dspark_noise_token_id as i32,
            g_zeros: None,
            g_main_hidden: None,
            g_pos_sp: None,
            g_pos_q: None,
            g_pos_kv: None,
            g_collapse_out: None,
            dgraphs: None,
            t_chain: 0.0,
            t_markov: 0.0,
            head_fp8: None,
            markov_w2_fp8: None,
            use_fp8_logits: false,
            g_graph_fp8_arm: false,
            g_logits_bf16: None,
        };
        me.maybe_make_fp8_heads()?;
        Ok(me)
    }

    /// --dspark-fp8-head / --dspark-fp8-logits: build fp8_bsb copies of the draft LM head
    /// [vocab, dim] and Markov W2 [vocab, rank] (halve the draft head reads — the vLLM stack's
    /// FP8 DeepGEMM copies, memo §2). Draft-side only: the trunk verify governs correctness;
    /// near-tie argmax flips cost acceptance, never correctness. The head's flag rides TpConfig
    /// (SPMD: both ranks build the same arms); the env var remains as the back-compat alias.
    fn maybe_make_fp8_heads(&mut self) -> Result<()> {
        let cfg_flag = crate::tp::tp_config().map(|c| c.dspark_fp8_head).unwrap_or(false);
        if !cfg_flag && !env_flag_once(crate::opt!("dspark-fp8-logits")) {
            return Ok(());
        }
        self.use_fp8_logits = true;
        let (vocab, dim, rank) = (self.cfg.vocab_size, self.cfg.dim, self.cfg.dspark_markov_rank);
        eprintln!("[dspark] fp8 draft head ON (--dspark-fp8-head / --dspark-fp8-logits): quantizing lm_head [{vocab},{dim}] + markov_w2 [{vocab},{rank}] → fp8_bsb ...");
        let t0 = std::time::Instant::now();
        let head_host: Vec<bf16> = self.dev.dtoh_sync_copy(&self.head)?;
        let (wt, sb) = quant::quantize_fp8_bsb(&head_host, vocab, dim);
        drop(head_host);
        self.head_fp8 = Some(Fp8Weight {
            wt: self.dev.htod_sync_copy(&wt)?,
            sb: self.dev.htod_sync_copy(&sb)?,
            m: vocab, k: dim,
        });
        let mw2_host: Vec<bf16> = self.dev.dtoh_sync_copy(&self.markov_w2)?;
        let (wt2, sb2) = quant::quantize_fp8_bsb(&mw2_host, vocab, rank);
        self.markov_w2_fp8 = Some(Fp8Weight {
            wt: self.dev.htod_sync_copy(&wt2)?,
            sb: self.dev.htod_sync_copy(&sb2)?,
            m: vocab, k: rank,
        });
        eprintln!("[dspark]   fp8 draft heads ready in {:.1}s (head 1060→530 MB, w2 66→33 MB per draft read)", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Warm all 3 stages' rings from the prefill main_hidden [s, 3*dim]. Computes main_x once,
    /// then each stage writes main_kv for all s positions (no attention/FFN — §B.10 warm branch).
    pub fn warm(&mut self, main_hidden: &B, s: usize) -> Result<()> {
        let main_x = self.main_x_proj(main_hidden, s)?;
        let rope = self.rt.rope_for(dsv4_load::LayerKind::Swa);
        for i in 0..self.stages.len() {
            self.rt
                .dspark_attn_warm(&self.stages[i], &mut self.states[i], &main_x, s, rope, &self.cfg)
                .with_context(|| format!("dspark warm stage {i}"))?;
        }
        Ok(())
    }

    /// Re-prime the draft rings for a RANGE of committed positions (the DSpark verify re-prime —
    /// §2.6 "re-prime with real verify hiddens"). `main_hidden` [s, 3*dim] is the trunk hidden
    /// mean for positions `start_pos .. start_pos+s-1` (captured from the verify forward). Each
    /// stage writes main_kv at those positions, keeping the draft ring contiguous (no gaps at the
    /// accepted-draft positions → the draft attends to real main_kv, not stale slots).
    pub fn warm_range(&mut self, main_hidden: &B, s: usize, start_pos: usize) -> Result<()> {
        if s == 0 { return Ok(()); }
        let main_x = self.main_x_proj(main_hidden, s)?;
        let rope = self.rt.rope_for(dsv4_load::LayerKind::Swa);
        for i in 0..self.stages.len() {
            self.rt
                .dspark_attn_warm_range(&self.stages[i], &mut self.states[i], &main_x, s, start_pos, rope, &self.cfg)
                .with_context(|| format!("dspark warm_range stage {i}"))?;
        }
        Ok(())
    }

    /// The draft forward. `main_hidden` [1, 3*dim] is the trunk hidden mean at `start_pos` (the
    /// last committed token's position). `real_token` is the trunk's argmax for start_pos+1.
    /// Returns `block` draft tokens + raw confidence.
    pub fn draft(&mut self, main_hidden: &B, real_token: i32, start_pos: usize) -> Result<DraftOut> {
        Ok(self.draft_full(main_hidden, real_token, start_pos)?.0)
    }

    /// The draft forward at a reduced width: draft `n` rows (1..=block) instead of `block`.
    /// Item 3.3 (adaptive draft depth): the policy truncates the parallel-row pass when the
    /// deep positions' acceptance does not pay their width-proportional MoE/projection bytes.
    /// Same kernels, same arithmetic, only the row count changes — a depth-D step is bitwise
    /// the fixed-depth step's first D rows (the trunk verify governs correctness either way).
    pub fn draft_n(&mut self, main_hidden: &B, real_token: i32, start_pos: usize, n: usize) -> Result<DraftOut> {
        Ok(self.draft_full_n(main_hidden, real_token, start_pos, n)?.0)
    }

    /// [`draft`](Self::draft) + the [block, vocab] fp32 logits (the bitwise graph gate
    /// compares them against the graphed arm's replay output).
    pub fn draft_full(&mut self, main_hidden: &B, real_token: i32, start_pos: usize) -> Result<(DraftOut, Vec<f32>)> {
        self.draft_full_n(main_hidden, real_token, start_pos, self.block)
    }

    /// [`draft_full`](Self::draft_full) at width `n` (see [`draft_n`](Self::draft_n)).
    fn draft_full_n(&mut self, main_hidden: &B, real_token: i32, start_pos: usize, n: usize) -> Result<(DraftOut, Vec<f32>)> {
        anyhow::ensure!(n >= 1 && n <= self.block, "draft_n width {n} out of range 1..={}", self.block);
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let eps = self.cfg.norm_eps;
        let phase = env_flag_once(crate::opt!("dspark-phase-ms"));
        let _tc = std::time::Instant::now();

        // 1. main_x = main_norm(main_proj(main_hidden)) [1, dim].
        let main_x = self.main_x_proj(main_hidden, 1)?;

        // 2. draft ids [real, noise×(n-1)] + embed → ×hc.
        let mut draft_ids = vec![self.noise_id; n];
        draft_ids[0] = real_token;
        let ids_dev = self.dev.htod_sync_copy(&draft_ids)?;
        let mut h = self.rt.embed_tokens(&self.embed, &ids_dev, n, &self.cfg)?;

        // 3. chain the 3 stages.
        for i in 0..self.stages.len() {
            h = self.dspark_block_forward(i, &h, n, start_pos, &main_x)?;
        }

        // 4. forward_head (stage 2): hc_head → norm → LM head (fp32) → [n, vocab].
        let collapse = self
            .rt
            .hc_head(&h, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, n, &self.cfg)?;
        let yn = self.rt.rmsnorm(&collapse, &self.stage2_norm, n, dim, eps)?;

        // 5+6. sequential Markov chain + confidence head (host tail, shared with the graphed path).
        if phase {
            unsafe { cudarc::driver::result::stream::synchronize(self.rt.stream.stream).ok() };
            self.t_chain += _tc.elapsed().as_secs_f64();
        }
        let _tm = std::time::Instant::now();
        // fp8 draft head (--dspark-fp8-logits): halves the 1.06 GB head read per draft.
        let logits_host: Vec<f32> = if self.use_fp8_logits && self.head_fp8.is_some() {
            let h8 = self.head_fp8.as_ref().unwrap();
            let (cc, csa) = self.rt.quant_g128::<B, CudaSlice<u8>>(&yn, n, dim)?;
            let lg = self.rt.fp8_bsb_rows::<B, CudaSlice<u8>>(h8, &cc, &csa, n)?;
            self.dev.dtoh_sync_copy(&lg)?.iter().map(|v| v.to_f32()).collect()
        } else {
            let logits = self.rt.lm_head(&self.head, &yn, n, dim, vocab)?; // [n, vocab] fp32
            self.dev.dtoh_sync_copy(&logits)?
        };
        let collapse_host: Vec<bf16> = self.dev.dtoh_sync_copy(&collapse)?;
        let out = self.markov_tail(&logits_host, &collapse_host, real_token, n)?;
        if phase {
            self.t_markov += _tm.elapsed().as_secs_f64();
        }
        Ok((out, logits_host))
    }

    /// The host tail of the draft: the sequential Markov bigram chain (greedy argmax over
    /// row + markov_w2@e bias) + the raw confidence head. Shared by [`draft`](Self::draft)
    /// (eager logits) and [`draft_graphed`](Self::draft_graphed) (graph-replayed logits) —
    /// identical arithmetic on identical inputs ⇒ identical draft tokens. `n` = draft rows.
    fn markov_tail(&self, logits_host: &[f32], collapse_host: &[bf16], real_token: i32, n: usize) -> Result<DraftOut> {
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        // 5. sequential Markov bigram chain (greedy: argmax). The chain is serial —
        //    output_ids[i+1] depends on the argmax after the i-th bias add. The bias GEMV runs
        //    on device (markov_w2 @ e); the add + argmax run on host (the row is dtoh'd).
        let rank = self.cfg.dspark_markov_rank;
        let mut output_ids = vec![0i32; n + 1];
        output_ids[0] = real_token;
        for i in 0..n {
            let id = output_ids[i] as usize;
            let e_host = &self.markov_w1_host[id * rank..(id + 1) * rank];
            let e_dev: B = self.dev.htod_sync_copy(e_host)?;
            // fp8 Markov W2 (--dspark-fp8-logits): halves the 66 MB w2 read per step.
            let bias_host: Vec<f32> = if self.use_fp8_logits && self.markov_w2_fp8.is_some() {
                let m8 = self.markov_w2_fp8.as_ref().unwrap();
                let (cc, csa) = self.rt.quant_g128::<B, CudaSlice<u8>>(&e_dev, 1, rank)?;
                let b = self.rt.fp8_bsb_rows::<B, CudaSlice<u8>>(m8, &cc, &csa, 1)?;
                self.dev.dtoh_sync_copy(&b)?.iter().map(|v| v.to_f32()).collect()
            } else {
                let bias = self.rt.lm_head(&self.markov_w2, &e_dev, 1, rank, vocab)?;
                self.dev.dtoh_sync_copy(&bias)?
            };
            let row = &logits_host[i * vocab..(i + 1) * vocab];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for v in 0..vocab {
                let lv = row[v] + bias_host[v];
                if lv > bv { bv = lv; best = v; }
            }
            output_ids[i + 1] = best as i32;
        }

        // 6. confidence head (raw fp32; logged, unused v1). cat([collapse, markov_embeds]).
        let mut confidence = vec![0.0f32; n];
        let conf_host: Vec<f32> = self.dev.dtoh_sync_copy(&self.confidence)?;
        for i in 0..n {
            let id = output_ids[i] as usize;
            let emb = &self.markov_w1_host[id * rank..(id + 1) * rank];
            let mut acc = 0.0f32;
            for d in 0..dim {
                acc += collapse_host[i * dim + d].to_f32() * conf_host[d];
            }
            for r in 0..rank {
                acc += emb[r].to_f32() * conf_host[dim + r];
            }
            confidence[i] = acc;
        }

        Ok(DraftOut { drafts: output_ids[1..=n].to_vec(), confidence })
    }

    /// CUDA-graph draft (--dspark-graph=1): the device chain (main_x → embed → 3 stages →
    /// hc_head → norm → LM head) replays a captured whole-chain graph instead of ~170 eager
    /// launches + per-call htod/iota uploads; the Markov tail stays eager (host-serial by
    /// design). Bitwise by construction (same kernels/args/order — the classifier verifies
    /// every position-dependent arg against its capture-time formula). Capture/classify
    /// errors poison the graph (eager forever after, loudly); a poisoned graph makes the
    /// caller's eager fallback always safe (capture records, never executes).
    pub fn draft_graphed(&mut self, main_hidden: &B, real_token: i32, start_pos: usize) -> Result<DraftOut> {
        Ok(self.draft_graphed_full(main_hidden, real_token, start_pos)?.0)
    }

    /// [`draft_graphed`](Self::draft_graphed) + the [block, vocab] fp32 logits (the gate).
    pub fn draft_graphed_full(&mut self, main_hidden: &B, real_token: i32, start_pos: usize) -> Result<(DraftOut, Vec<f32>)> {
        if !env_flag_once(crate::opt!("dspark-graph")) {
            return self.draft_full(main_hidden, real_token, start_pos);
        }
        self.ensure_graph_inputs()?;
        if self.dgraphs.is_none() {
            let mut func_names: std::collections::HashMap<usize, &'static str> = std::collections::HashMap::new();
            for (n, f) in self.rt.spine.func_handles() {
                func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
            }
            for (n, f) in self.rt.attn.func_handles() {
                func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
            }
            crate::dsv4_graph::raise_mempool_threshold(&self.dev)?;
            let _ = crate::dsv4_gpu::graph_mempool(&self.dev);
            // Shares the global workspace slab with the trunk graphs (idempotent init):
            // transients never persist across a replay, so region overlap is sound.
            crate::dsv4_gpu::graph_ws_init(&self.dev, 256 * 1024 * 1024)?;
            self.dgraphs = Some(DecodeGraphs::new_drafter(
                &self.dev, func_names, self.rt.window, self.block, self.cfg.vocab_size,
            )?);
        }
        let mut graphs = self.dgraphs.take().unwrap();
        let result = self.draft_graphed_step(&mut graphs, main_hidden, real_token, start_pos);
        self.dgraphs = Some(graphs);
        result
    }

    /// Allocate the persistent graph inputs once (zeros via htod — `alloc_zeros` does NOT
    /// zero, AGENTS §2.2; the contents of the rest are refreshed per step before replay).
    fn ensure_graph_inputs(&mut self) -> Result<()> {
        if self.g_zeros.is_some() {
            return Ok(());
        }
        let (dim, block) = (self.cfg.dim, self.block);
        let nh = self.cfg.n_heads;
        self.g_zeros = Some(self.dev.htod_sync_copy(&vec![0i32; block])?);
        self.g_main_hidden = Some(self.dev.htod_sync_copy(&vec![bf16::ZERO; 3 * dim])?);
        self.g_pos_sp = Some(self.dev.htod_sync_copy(&vec![0i32; 1])?);
        self.g_pos_q = Some(self.dev.htod_sync_copy(&vec![0i32; block * nh])?);
        self.g_pos_kv = Some(self.dev.htod_sync_copy(&vec![0i32; block])?);
        self.g_collapse_out = Some(self.dev.htod_sync_copy(&vec![bf16::ZERO; block * dim])?);
        self.g_logits_bf16 = Some(self.dev.htod_sync_copy(&vec![bf16::ZERO; block * self.cfg.vocab_size])?);
        Ok(())
    }

    /// Refresh the persistent graph inputs for this step (stream-ordered, NO host sync):
    /// draft ids htod, main_hidden d2d, and the three position buffers via `dsv4_iota_b`
    /// (out[i] = start + (i*mul)/div — the same integer math the eager impl's iota used).
    fn refresh_graph_inputs(&self, graphs: &DecodeGraphs, main_hidden: &B, real_token: i32, start_pos: usize) -> Result<()> {
        let block = self.block;
        let nh = self.cfg.n_heads;
        let mut draft_ids = vec![self.noise_id; block];
        draft_ids[0] = real_token;
        let stream = self.rt.stream.stream;
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                *graphs.ids_dev.device_ptr(), &draft_ids, stream,
            )
            .map_err(|e| anyhow!("dspark-graph ids htod: {e}"))?;
            cudarc::driver::result::memcpy_dtod_async(
                *self.g_main_hidden.as_ref().unwrap().device_ptr(),
                *main_hidden.device_ptr(),
                3 * self.cfg.dim * 2,
                stream,
            )
            .map_err(|e| anyhow!("dspark-graph main_hidden d2d: {e}"))?;
        }
        let pos0 = (start_pos + 1) as i32;
        let sp_i = start_pos as i32;
        let (zero_i, one_i, nh_i) = (0i32, 1i32, nh as i32);
        let (rows_q_i, blk_i) = ((block * nh) as i32, block as i32);
        let pos_sp = self.g_pos_sp.as_ref().unwrap();
        let pos_q = self.g_pos_q.as_ref().unwrap();
        let pos_kv = self.g_pos_kv.as_ref().unwrap();
        dsv4_launch!(self.rt.spine, "dsv4_iota_b", stream, (1u32, 1, 1), (256, 1, 1), 0,
            (pos_sp, &sp_i, &zero_i, &one_i, &one_i))?;
        dsv4_launch!(self.rt.spine, "dsv4_iota_b", stream, (((block * nh + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (pos_q, &pos0, &one_i, &nh_i, &rows_q_i))?;
        dsv4_launch!(self.rt.spine, "dsv4_iota_b", stream, (1u32, 1, 1), (256, 1, 1), 0,
            (pos_kv, &pos0, &one_i, &one_i, &blk_i))?;
        Ok(())
    }

    fn draft_graphed_step(
        &mut self,
        graphs: &mut DecodeGraphs,
        main_hidden: &B,
        real_token: i32,
        start_pos: usize,
    ) -> Result<(DraftOut, Vec<f32>)> {
        if matches!(graphs.slot_ref(Variant::V0), Slot::Poisoned) {
            return Err(anyhow!("drafter graph poisoned (earlier capture failure)"));
        }
        // Refresh inputs BEFORE capture (the capture records buffer ADDRESSES; the replay
        // that follows reads the same contents — capture itself executes nothing).
        self.refresh_graph_inputs(graphs, main_hidden, real_token, start_pos)?;
        if matches!(graphs.slot_ref(Variant::V0), Slot::Unborn) {
            let cap_result = (|| {
                let r = unsafe {
                    cudarc::driver::sys::cuStreamBeginCapture_v2(
                        self.rt.stream.stream,
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                };
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("dspark cuStreamBeginCapture: {r:?}"));
                }
                crate::dsv4_gpu::GRAPH_CAPTURE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
                crate::dsv4_gpu::graph_ws_begin_capture();
                // ONE slab-wide memset as the graph's first node (the trunk-graph pattern).
                unsafe {
                    let (d, b) = crate::dsv4_gpu::graph_ws_span();
                    cudarc::driver::result::memset_d8_async(d, 0, b, self.rt.stream.stream)
                        .map_err(|e| anyhow!("dspark graph slab memset: {e}"))?;
                }
                let fwd = (|| {
                    let (logits, collapse) = self.draft_chain_dev(&graphs.ids_dev, start_pos)?;
                    // Outputs must land in persistent eager buffers INSIDE the graph
                    // (memcpy nodes — in-graph slab memory must never be read directly).
                    unsafe {
                        match &logits {
                            DraftLogitsDev::F32(lg) => {
                                cudarc::driver::result::memcpy_dtod_async(
                                    *graphs.logits_out.device_ptr(),
                                    lg.dptr(),
                                    lg.len() * 4,
                                    self.rt.stream.stream,
                                )
                                .map_err(|e| anyhow!("dspark graph logits-out memcpy: {e}"))?;
                            }
                            DraftLogitsDev::BF16(lg) => {
                                cudarc::driver::result::memcpy_dtod_async(
                                    *self.g_logits_bf16.as_ref().unwrap().device_ptr(),
                                    lg.dptr(),
                                    lg.len() * 2,
                                    self.rt.stream.stream,
                                )
                                .map_err(|e| anyhow!("dspark graph logits-bf16-out memcpy: {e}"))?;
                            }
                        }
                        cudarc::driver::result::memcpy_dtod_async(
                            *self.g_collapse_out.as_ref().unwrap().device_ptr(),
                            collapse.dptr(),
                            collapse.len() * 2,
                            self.rt.stream.stream,
                        )
                        .map_err(|e| anyhow!("dspark graph collapse-out memcpy: {e}"))?;
                    }
                    Ok::<_, anyhow::Error>(())
                })();
                let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
                let r = unsafe { cudarc::driver::sys::cuStreamEndCapture(self.rt.stream.stream, &mut graph) };
                crate::dsv4_gpu::GRAPH_CAPTURE_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("dspark cuStreamEndCapture: {r:?}"));
                }
                fwd?;
                let g = graphs.instantiate(graph, start_pos)?;
                g.upload_once(&self.rt.stream)?;
                eprintln!("[dspark-graph] captured at sp={start_pos} (ws high-water {} MB, fp8_logits={})",
                    crate::dsv4_gpu::graph_ws_high_water() / (1024 * 1024), self.use_fp8_logits);
                *graphs.slot_mut(Variant::V0) = Slot::Ready(g);
                self.g_graph_fp8_arm = self.use_fp8_logits;
                Ok::<_, anyhow::Error>(())
            })();
            if let Err(e) = cap_result {
                *graphs.slot_mut(Variant::V0) = Slot::Poisoned;
                return Err(e);
            }
        }
        let g = match graphs.slot_ref(Variant::V0) {
            Slot::Ready(g) => g,
            _ => unreachable!("post-capture slot is Ready"),
        };
        if self.use_fp8_logits != self.g_graph_fp8_arm {
            return Err(anyhow!(
                "use_fp8_logits flipped after the draft graph was captured (arm={}) — the graph bakes one arm's output copy",
                self.g_graph_fp8_arm
            ));
        }
        g.apply_updates_and_launch(&self.rt.stream, start_pos)?;
        let phase = env_flag_once(crate::opt!("dspark-phase-ms"));
        let _tm = std::time::Instant::now();
        let logits_host: Vec<f32> = if self.use_fp8_logits {
            let lg: Vec<bf16> = self.dev.dtoh_sync_copy(self.g_logits_bf16.as_ref().unwrap())?;
            if phase {
                self.t_chain += _tm.elapsed().as_secs_f64();
            }
            lg.iter().map(|v| v.to_f32()).collect()
        } else {
            let lg = self.dev.dtoh_sync_copy(&graphs.logits_out)?;
            if phase {
                self.t_chain += _tm.elapsed().as_secs_f64();
            }
            lg
        };
        let _tm2 = std::time::Instant::now();
        let collapse_host: Vec<bf16> = self.dev.dtoh_sync_copy(self.g_collapse_out.as_ref().unwrap())?;
        let out = self.markov_tail(&logits_host, &collapse_host, real_token, self.block)?;
        if phase {
            self.t_markov += _tm2.elapsed().as_secs_f64();
        }
        Ok((out, logits_host))
    }

    /// The graphed device chain: main_x → embed → 3 stages → hc_head → norm → LM head.
    /// Returns (logits [block, vocab], collapse [block, dim] bf16) as workspace-slab
    /// slices — valid only until the next replay; the caller memcpy's them out in-graph.
    fn draft_chain_dev(&mut self, ids_dev: &CudaSlice<i32>, start_pos: usize) -> Result<(DraftLogitsDev, GB)> {
        let (dim, block, vocab) = (self.cfg.dim, self.block, self.cfg.vocab_size);
        let eps = self.cfg.norm_eps;
        let main_x = self.main_x_proj_dev::<GB, crate::dsv4_gpu::GSlice<u8>>(self.g_main_hidden.as_ref().unwrap(), 1)?;
        let mut h = self.rt.embed_tokens::<GB>(&self.embed, ids_dev, block, &self.cfg)?;
        for i in 0..self.stages.len() {
            h = self.dspark_block_forward_dev(i, &h, block, start_pos, &main_x)?;
        }
        let collapse = self
            .rt
            .hc_head::<GB>(&h, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, block, &self.cfg)?;
        let yn = self.rt.rmsnorm(&collapse, &self.stage2_norm, block, dim, eps)?;
        let logits = if self.use_fp8_logits && self.head_fp8.is_some() {
            let h8 = self.head_fp8.as_ref().unwrap();
            let (cc, csa) = self.rt.quant_g128::<GB, crate::dsv4_gpu::GSlice<u8>>(&yn, block, dim)?;
            DraftLogitsDev::BF16(self.rt.fp8_bsb_rows::<GB, crate::dsv4_gpu::GSlice<u8>>(h8, &cc, &csa, block)?)
        } else {
            DraftLogitsDev::F32(self.rt.lm_head::<GB, GS>(&self.head, &yn, block, dim, vocab)?)
        };
        Ok((logits, collapse))
    }

    /// GSlice-instantiated `main_x_proj` (quant codes ride the workspace slab under capture).
    fn main_x_proj_dev<X: crate::dsv4_gpu::Dsv4Buf<bf16>, C: crate::dsv4_gpu::Dsv4Buf<u8>>(&self, main_hidden: &B, s: usize) -> Result<X> {
        let (dim, eps) = (self.cfg.dim, self.cfg.norm_eps);
        let three_d = 3 * dim;
        let (cc, csa) = self.rt.quant_g128::<B, C>(main_hidden, s, three_d)?;
        let mx = self.rt.fp8_bsb_rows::<X, C>(&self.main_proj, &cc, &csa, s)?;
        self.rt.rmsnorm(&mx, &self.main_norm, s, dim, eps)
    }

    /// GSlice-instantiated `dspark_block_forward` (the graphed twin — same kernel sequence).
    fn dspark_block_forward_dev(
        &mut self,
        i: usize,
        x: &GB,
        block: usize,
        start_pos: usize,
        main_x: &GB,
    ) -> Result<GB> {
        let (dim, eps) = (self.cfg.dim, self.cfg.norm_eps);
        let layer = &self.stages[i];
        let rope = self.rt.rope_for(dsv4_load::LayerKind::Swa);
        let (y, posts, combs) = self
            .rt
            .hc_pre::<GB, GS>(x, block, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, &self.cfg)?;
        let yn = self.rt.rmsnorm(&y, &layer.attn_norm, block, dim, eps)?;
        let attn_out = self.rt.dspark_attn_forward_dev::<GB, crate::dsv4_gpu::GSlice<u8>, crate::dsv4_gpu::GSlice<i32>>(
            layer, &mut self.states[i], &yn, block, start_pos, main_x, rope, &self.cfg,
            self.g_pos_sp.as_ref().unwrap(),
            self.g_pos_q.as_ref().unwrap(),
            self.g_pos_kv.as_ref().unwrap(),
        )?;
        let x2 = self.rt.hc_post(&attn_out, x, &posts, &combs, block, &self.cfg)?;
        let (y2, posts2, combs2) = self
            .rt
            .hc_pre::<GB, GS>(&x2, block, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, &self.cfg)?;
        let y2n = self.rt.rmsnorm(&y2, &layer.ffn_norm, block, dim, eps)?;
        let (ffn_out, _rw, _ri) = self.rt.moe_forward::<GB, GS, crate::dsv4_gpu::GSlice<i32>>(
            layer, &mut self.scratch, &y2n, block, self.g_zeros.as_ref().unwrap(), &self.cfg)?;
        self.rt.hc_post(&ffn_out, &x2, &posts2, &combs2, block, &self.cfg)
    }

    /// R4 d3-cliff audit: mirrors `draft` but captures the per-stage intermediates so the
    /// probe can bisect where the chain diverges from the reference oracle
    /// (draft.h_in / h0 / h1 / h2 / logits in dsv4_dspark.npz). Debug-only; the serving
    /// path uses `draft`.
    pub fn draft_capture(
        &mut self,
        main_hidden: &B,
        real_token: i32,
        start_pos: usize,
    ) -> Result<(DraftOut, Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>, Vec<f32>)> {
        let (dim, block, vocab) = (self.cfg.dim, self.block, self.cfg.vocab_size);
        let eps = self.cfg.norm_eps;
        let dev = self.dev.clone();
        let to_f32 = move |b: &B| -> Result<Vec<f32>> {
            Ok(dev.dtoh_sync_copy(b)?.iter().map(|v| v.to_f32()).collect())
        };
        let main_x = self.main_x_proj(main_hidden, 1)?;
        let mut draft_ids = vec![self.noise_id; block];
        draft_ids[0] = real_token;
        let ids_dev = self.dev.htod_sync_copy(&draft_ids)?;
        let mut h = self.rt.embed_tokens(&self.embed, &ids_dev, block, &self.cfg)?;
        let mut stages = vec![to_f32(&h)?];
        let mut sublayers: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        for i in 0..self.stages.len() {
            let (h2, yn_host, attn_host, ffn_host, o_host, oflat_host, q_pre_host, q_rsc_host, kv_host) = self.dspark_block_forward_capture(i, &h, block, start_pos, &main_x)?;
            h = h2;
            stages.push(to_f32(&h)?);
            sublayers.push((yn_host, attn_host, ffn_host, o_host, oflat_host, q_pre_host, q_rsc_host, kv_host));
        }
        let collapse = self
            .rt
            .hc_head(&h, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, block, &self.cfg)?;
        let yn = self.rt.rmsnorm(&collapse, &self.stage2_norm, block, dim, eps)?;
        let logits = self.rt.lm_head(&self.head, &yn, block, dim, vocab)?;
        let logits_host: Vec<f32> = self.dev.dtoh_sync_copy(&logits)?;
        let rank = self.cfg.dspark_markov_rank;
        let mut output_ids = vec![0i32; block + 1];
        output_ids[0] = real_token;
        for i in 0..block {
            let id = output_ids[i] as usize;
            let e_host = &self.markov_w1_host[id * rank..(id + 1) * rank];
            let e_dev: B = self.dev.htod_sync_copy(e_host)?;
            let bias = self.rt.lm_head(&self.markov_w2, &e_dev, 1, rank, vocab)?;
            let bias_host: Vec<f32> = self.dev.dtoh_sync_copy(&bias)?;
            let row = &logits_host[i * vocab..(i + 1) * vocab];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for v in 0..vocab {
                let lv = row[v] + bias_host[v];
                if lv > bv { bv = lv; best = v; }
            }
            output_ids[i + 1] = best as i32;
        }
        let confidence = vec![0.0f32; block];
        Ok((DraftOut { drafts: output_ids[1..=block].to_vec(), confidence }, stages, sublayers, logits_host))
    }

    /// main_x = rmsnorm(main_proj @ main_hidden, main_norm). main_proj [dim, 3*dim] FP8.
    fn main_x_proj(&self, main_hidden: &B, s: usize) -> Result<B> {
        let (dim, eps) = (self.cfg.dim, self.cfg.norm_eps);
        let three_d = 3 * dim;
        let (cc, csa) = self.rt.quant_g128::<B, CudaSlice<u8>>(main_hidden, s, three_d)?;
        let mx = self.rt.fp8_bsb_rows(&self.main_proj, &cc, &csa, s)?;
        self.rt.rmsnorm(&mx, &self.main_norm, s, dim, eps)
    }

    /// Debug (R4 audit): main_x as host f32 for A/B against the CPU reference.
    pub fn main_x_for_debug(&self, main_hidden: &B, s: usize) -> Result<Vec<f32>> {
        let mx = self.main_x_proj(main_hidden, s)?;
        Ok(self.dev.dtoh_sync_copy(&mx)?.iter().map(|v| v.to_f32()).collect())
    }

    /// DSpark block forward: hc_pre → norm → DSparkAttention → hc_post → hc_pre → norm → MoE →
    /// hc_post. Same structure as trunk block_forward but the attention is the DSpark variant.
    /// Single-process MoE (no TP all-reduce).
    fn dspark_block_forward(
        &mut self,
        i: usize,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
    ) -> Result<B> {
        let (dim, eps) = (self.cfg.dim, self.cfg.norm_eps);
        let layer = &self.stages[i];
        let rope = self.rt.rope_for(dsv4_load::LayerKind::Swa);
        let (y, posts, combs) = self
            .rt
            .hc_pre::<B, S>(x, block, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, &self.cfg)?;
        let yn = self.rt.rmsnorm(&y, &layer.attn_norm, block, dim, eps)?;
        let attn_out = self
            .rt
            .dspark_attn_forward(layer, &mut self.states[i], &yn, block, start_pos, main_x, rope, &self.cfg)?;
        let x2 = self.rt.hc_post(&attn_out, x, &posts, &combs, block, &self.cfg)?;
        let (y2, posts2, combs2) = self
            .rt
            .hc_pre::<B, S>(&x2, block, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, &self.cfg)?;
        let y2n = self.rt.rmsnorm(&y2, &layer.ffn_norm, block, dim, eps)?;
        let ids = self.dev.htod_sync_copy(&vec![0i32; block])?;
        let (ffn_out, _rw, _ri) = self.rt.moe_forward::<B, S, CudaSlice<i32>>(layer, &mut self.scratch, &y2n, block, &ids, &self.cfg)?;
        self.rt.hc_post(&ffn_out, &x2, &posts2, &combs2, block, &self.cfg)
    }

    /// Debug (R4 audit): mirrors `dspark_block_forward` with sublayer captures
    /// (attn_out, ffn_out) as host f32 — splits attention vs MoE for the oracle bisect.
    pub fn dspark_block_forward_capture(
        &mut self,
        i: usize,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
    ) -> Result<(B, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (dim, eps) = (self.cfg.dim, self.cfg.norm_eps);
        let layer = &self.stages[i];
        let rope = self.rt.rope_for(dsv4_load::LayerKind::Swa);
        let (y, posts, combs) = self
            .rt
            .hc_pre::<B, S>(x, block, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, &self.cfg)?;
        let yn = self.rt.rmsnorm(&y, &layer.attn_norm, block, dim, eps)?;
        let yn_host: Vec<f32> = self.dev.dtoh_sync_copy(&yn)?.iter().map(|v| v.to_f32()).collect();
        let attn_out = self
            .rt
            .dspark_attn_forward_capture(layer, &mut self.states[i], &yn, block, start_pos, main_x, rope, &self.cfg)?;
        let (attn_out, q_pre_host, q_rsc_host, kv_host, o_host, oflat_host) = attn_out;
        let attn_host: Vec<f32> = self.dev.dtoh_sync_copy(&attn_out)?.iter().map(|v| v.to_f32()).collect();
        let x2 = self.rt.hc_post(&attn_out, x, &posts, &combs, block, &self.cfg)?;
        let (y2, posts2, combs2) = self
            .rt
            .hc_pre::<B, S>(&x2, block, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, &self.cfg)?;
        let y2n = self.rt.rmsnorm(&y2, &layer.ffn_norm, block, dim, eps)?;
        let ids = self.dev.htod_sync_copy(&vec![0i32; block])?;
        let (ffn_out, _rw, _ri) = self.rt.moe_forward::<B, S, CudaSlice<i32>>(layer, &mut self.scratch, &y2n, block, &ids, &self.cfg)?;
        let ffn_host: Vec<f32> = self.dev.dtoh_sync_copy(&ffn_out)?.iter().map(|v| v.to_f32()).collect();
        let out = self.rt.hc_post(&ffn_out, &x2, &posts2, &combs2, block, &self.cfg)?;
        Ok((out, yn_host, attn_host, ffn_host, o_host, oflat_host, q_pre_host, q_rsc_host, kv_host))
    }
}

// ---- extras extraction helpers ----

fn take_f32_extra(
    dev: &Arc<CudaDevice>,
    map: &mut std::collections::HashMap<String, HostTensor>,
    key: &str,
    n: usize,
) -> Result<S> {
    let data = take_f32_data(map, key, n)?;
    Ok(dev.htod_sync_copy(&data)?)
}

fn take_f32_data(
    map: &mut std::collections::HashMap<String, HostTensor>,
    key: &str,
    n: usize,
) -> Result<Vec<f32>> {
    match map.remove(key) {
        Some(HostTensor::F32 { data, .. }) => {
            anyhow::ensure!(data.len() == n, "{key}: expected {n} f32, got {}", data.len());
            Ok(data)
        }
        other => Err(anyhow!("{key}: expected F32 extra, got {:?}", other.map(|t| t.shape().to_vec()))),
    }
}

fn take_bf16_extra(
    map: &mut std::collections::HashMap<String, HostTensor>,
    key: &str,
    n: usize,
) -> Result<Vec<bf16>> {
    match map.remove(key) {
        Some(HostTensor::BF16 { data, .. }) => {
            anyhow::ensure!(data.len() == n, "{key}: expected {n} bf16, got {}", data.len());
            Ok(data)
        }
        other => Err(anyhow!("{key}: expected BF16 extra, got {:?}", other.map(|t| t.shape().to_vec()))),
    }
}
