//! Declarative `ShardSpecPlan` — replaces hundreds of lines of per-model TP weight-slicing boilerplate
//! with a compact table of tensor name patterns -> sharding specs, aligned to hardware / quantization
//! boundaries (e.g. 128-element EXL3 trellis blocks).
//!
//! Includes ready-to-use presets:
//! - [`ShardSpecPlan::dense_transformer`]: standard Llama / Qwen / Gemma dense GQA + SwiGLU MLP.
//! - [`ShardSpecPlan::inkling2_default`]: tailored preset for Inkling-2 (GQA attention, FFN/MoE,
//!   vocab-parallel `lm_head`, 128-element EXL3 alignment).
//! - [`ShardSpecPlan::moe_hybrid`]: MoE models supporting either expert-parallel (`e % world == rank`)
//!   or expert-TP (every expert sharded on its inner width, like DSV4.1 / Naive).

use super::placement::{aligned_shard_range, Placement, ShardRange};
use anyhow::{bail, ensure, Result};

/// Simple, dependency-free tensor name matcher (no regex bloat).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    Exact(String),
    Suffix(String),
    Contains(String),
    /// Matches any tensor containing BOTH `a` and `b` (e.g. `".experts."` and `".down_proj"`).
    ContainsBoth(String, String),
}

impl Pattern {
    pub fn exact(s: impl Into<String>) -> Self { Pattern::Exact(s.into()) }
    pub fn suffix(s: impl Into<String>) -> Self { Pattern::Suffix(s.into()) }
    pub fn contains(s: impl Into<String>) -> Self { Pattern::Contains(s.into()) }
    pub fn contains_both(a: impl Into<String>, b: impl Into<String>) -> Self {
        Pattern::ContainsBoth(a.into(), b.into())
    }

    pub fn matches(&self, name: &str) -> bool {
        match self {
            Pattern::Exact(s) => name == s,
            Pattern::Suffix(s) => name.ends_with(s),
            Pattern::Contains(s) => name.contains(s),
            Pattern::ContainsBoth(a, b) => name.contains(a) && name.contains(b),
        }
    }
}

/// How a matched weight tensor is partitioned across `world` ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorSpec {
    /// Held whole on every rank (norms, embeddings, routers, biases).
    Replicate,
    /// Column-parallel projection weight `[out, in]`: shard dim 0 (output features) on `align` boundary.
    ShardCol { align: usize },
    /// Row-parallel projection weight `[out, in]`: shard dim 1 (input features) on `align` boundary.
    ShardRow { align: usize },
    /// Shard an arbitrary dimension `axis` on `align` boundary (e.g. 1-D biases, KV scales).
    ShardAxis { axis: usize, align: usize },
    /// Expert-parallel ownership: rank `r` owns expert `e` iff `e % world == r` (whole expert on its owner).
    ExpertParallel { num_experts: usize },
}

/// One ordered rule in a [`ShardSpecPlan`] (first matching rule wins).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub pattern: Pattern,
    pub spec: TensorSpec,
}

/// Concrete per-rank slicing decision returned by [`ShardSpecPlan::resolve`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedSlice {
    pub placement: Placement,
    /// `false` if an expert-parallel tensor belongs to another rank and should be skipped on this rank.
    pub owned: bool,
    /// Element range `[start, end)` along the sharded axis (`None` when `Replicated` or not owned).
    pub range: Option<ShardRange>,
    /// Resulting local tensor shape on this rank (empty if `!owned`).
    pub local_shape: Vec<usize>,
}

/// Declarative per-model sharding plan.
#[derive(Clone, Debug)]
pub struct ShardSpecPlan {
    pub family: String,
    pub rules: Vec<Rule>,
    pub default_spec: TensorSpec,
}

impl ShardSpecPlan {
    pub fn new(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            rules: Vec::new(),
            default_spec: TensorSpec::Replicate,
        }
    }

    pub fn rule(mut self, pattern: Pattern, spec: TensorSpec) -> Self {
        self.rules.push(Rule { pattern, spec });
        self
    }

    pub fn with_default(mut self, spec: TensorSpec) -> Self {
        self.default_spec = spec;
        self
    }

    /// Look up the [`TensorSpec`] for a tensor name (first matching rule wins, else `default_spec`).
    pub fn spec_for(&self, tensor_name: &str) -> TensorSpec {
        for r in &self.rules {
            if r.pattern.matches(tensor_name) {
                return r.spec;
            }
        }
        self.default_spec
    }

    /// Resolve a tensor's exact placement, ownership, axis range, and local shape for `(rank, world)`.
    pub fn resolve(
        &self,
        tensor_name: &str,
        global_shape: &[usize],
        rank: usize,
        world: usize,
    ) -> Result<ResolvedSlice> {
        ensure!(world >= 1 && rank < world, "ShardSpecPlan::resolve: invalid rank {rank}/{world}");
        if world == 1 {
            return Ok(ResolvedSlice {
                placement: Placement::Replicated,
                owned: true,
                range: None,
                local_shape: global_shape.to_vec(),
            });
        }
        let spec = self.spec_for(tensor_name);
        match spec {
            TensorSpec::Replicate => Ok(ResolvedSlice {
                placement: Placement::Replicated,
                owned: true,
                range: None,
                local_shape: global_shape.to_vec(),
            }),
            TensorSpec::ShardCol { align } => self.slice_axis(tensor_name, global_shape, 0, align, rank, world),
            TensorSpec::ShardRow { align } => {
                ensure!(
                    global_shape.len() >= 2,
                    "{tensor_name}: ShardRow requires >= 2D tensor, got {global_shape:?}"
                );
                self.slice_axis(tensor_name, global_shape, 1, align, rank, world)
            }
            TensorSpec::ShardAxis { axis, align } => {
                self.slice_axis(tensor_name, global_shape, axis, align, rank, world)
            }
            TensorSpec::ExpertParallel { num_experts } => {
                let eid = parse_expert_id(tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("{tensor_name}: ExpertParallel rule matched but no '.experts.<id>.' found")
                })?;
                ensure!(eid < num_experts, "{tensor_name}: expert id {eid} >= num_experts {num_experts}");
                let owned = (eid % world) == rank;
                Ok(ResolvedSlice {
                    placement: Placement::Replicated,
                    owned,
                    range: None,
                    local_shape: if owned { global_shape.to_vec() } else { Vec::new() },
                })
            }
        }
    }

    fn slice_axis(
        &self,
        name: &str,
        global_shape: &[usize],
        axis: usize,
        align: usize,
        rank: usize,
        world: usize,
    ) -> Result<ResolvedSlice> {
        if axis >= global_shape.len() {
            bail!("{name}: shard axis {axis} out of bounds for shape {global_shape:?}");
        }
        let range = aligned_shard_range(global_shape[axis], rank, world, align)?;
        let mut local_shape = global_shape.to_vec();
        local_shape[axis] = range.len();
        Ok(ResolvedSlice {
            placement: Placement::Shard { axis },
            owned: true,
            range: Some(range),
            local_shape,
        })
    }

    /// Standard dense transformer preset (Llama / Qwen / Gemma): ~10 declarative rules.
    pub fn dense_transformer(align: usize) -> Self {
        Self::new("dense_transformer")
            .rule(Pattern::contains("q_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("k_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("v_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("o_proj"), TensorSpec::ShardRow { align })
            .rule(Pattern::contains("gate_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("up_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("down_proj"), TensorSpec::ShardRow { align })
            .rule(Pattern::contains("lm_head"), TensorSpec::ShardCol { align })
            .with_default(TensorSpec::Replicate)
    }

    /// Inkling-2 declarative v2 preset: GQA attention (`q/k/v` col-sharded, `o` row-sharded),
    /// SwiGLU / MoE experts sharded on inner width (expert-TP, zero cross-rank dispatch),
    /// vocab-parallel `lm_head`, and replicated norms/embeddings/routers — all aligned to `align`
    /// (typically `128` for EXL3 trellis blocks).
    pub fn inkling2_default(align: usize) -> Self {
        Self::new("inkling2")
            // Norms, embeddings, routers stay replicated
            .rule(Pattern::contains("norm"), TensorSpec::Replicate)
            .rule(Pattern::contains("embed_tokens"), TensorSpec::Replicate)
            .rule(Pattern::contains("router"), TensorSpec::Replicate)
            .rule(Pattern::suffix(".gate.weight"), TensorSpec::Replicate)
            // Attention projections
            .rule(Pattern::contains("q_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("k_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("v_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("o_proj"), TensorSpec::ShardRow { align })
            // Dense & MoE FFN projections (expert-TP: shard inner width on every rank)
            .rule(Pattern::contains("gate_up_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("gate_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("up_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("down_proj"), TensorSpec::ShardRow { align })
            // Vocab-parallel output head
            .rule(Pattern::contains("lm_head"), TensorSpec::ShardCol { align })
            .with_default(TensorSpec::Replicate)
    }

    /// Hybrid MoE preset: attention is TP-sharded, routed experts are either expert-TP (`expert_tp = true`,
    /// like DSV4.1/Naive) or expert-parallel (`expert_tp = false`, like EXL3 Qwen3-MoE), and shared experts
    /// are TP-sharded on inner width.
    pub fn moe_hybrid(num_experts: usize, expert_tp: bool, align: usize) -> Self {
        let mut p = Self::new("moe_hybrid")
            .rule(Pattern::contains("norm"), TensorSpec::Replicate)
            .rule(Pattern::contains("embed_tokens"), TensorSpec::Replicate)
            .rule(Pattern::contains("q_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("k_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("v_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("o_proj"), TensorSpec::ShardRow { align });
        if expert_tp {
            p = p
                .rule(Pattern::contains_both(".experts.", "gate_proj"), TensorSpec::ShardCol { align })
                .rule(Pattern::contains_both(".experts.", "up_proj"), TensorSpec::ShardCol { align })
                .rule(Pattern::contains_both(".experts.", "down_proj"), TensorSpec::ShardRow { align });
        } else {
            p = p.rule(Pattern::contains(".experts."), TensorSpec::ExpertParallel { num_experts });
        }
        p.rule(Pattern::contains("shared_expert"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("gate_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("up_proj"), TensorSpec::ShardCol { align })
            .rule(Pattern::contains("down_proj"), TensorSpec::ShardRow { align })
            .rule(Pattern::contains("lm_head"), TensorSpec::ShardCol { align })
            .with_default(TensorSpec::Replicate)
    }
}

/// Extract numeric expert ID from HuggingFace-style names like `model.layers.3.mlp.experts.42.down_proj.weight`.
fn parse_expert_id(name: &str) -> Option<usize> {
    let rest = name.split(".experts.").nth(1)?;
    let tok = rest.split('.').next()?;
    tok.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inkling2_preset_shards_attention_mlp_and_vocab_on_128_alignment() {
        let plan = ShardSpecPlan::inkling2_default(128);
        // Col-parallel Q proj [4096, 4096] across TP=4 -> [1024, 4096]
        let q = plan.resolve("model.layers.0.self_attn.q_proj.weight", &[4096, 4096], 2, 4).unwrap();
        assert_eq!(q.placement, Placement::Shard { axis: 0 });
        assert_eq!(q.local_shape, vec![1024, 4096]);
        assert_eq!(q.range, Some(ShardRange { start: 2048, end: 3072 }));
        // Row-parallel O proj [4096, 4096] across TP=4 -> [4096, 1024]
        let o = plan.resolve("model.layers.0.self_attn.o_proj.weight", &[4096, 4096], 1, 4).unwrap();
        assert_eq!(o.placement, Placement::Shard { axis: 1 });
        assert_eq!(o.local_shape, vec![4096, 1024]);
        assert_eq!(o.range, Some(ShardRange { start: 1024, end: 2048 }));
        // Router gate stays replicated (not caught by gate_proj!)
        let g = plan.resolve("model.layers.0.mlp.gate.weight", &[128, 4096], 3, 4).unwrap();
        assert_eq!(g.placement, Placement::Replicated);
        assert_eq!(g.local_shape, vec![128, 4096]);
        // Expert MLP inner width sharded on dim 0 / dim 1
        let eg = plan.resolve("model.layers.0.mlp.experts.7.gate_proj.weight", &[2048, 4096], 0, 4).unwrap();
        assert_eq!(eg.local_shape, vec![512, 4096]);
        let ed = plan.resolve("model.layers.0.mlp.experts.7.down_proj.weight", &[4096, 2048], 3, 4).unwrap();
        assert_eq!(ed.local_shape, vec![4096, 512]);
        // Vocab-parallel lm_head [129280, 4096] -> exact 128-aligned tiling across TP=4
        let v_sum: usize = (0..4)
            .map(|r| plan.resolve("lm_head.weight", &[129_280, 4096], r, 4).unwrap().local_shape[0])
            .sum();
        assert_eq!(v_sum, 129_280);
        // Solo TP=1: everything replicated, zero slicing
        let solo = plan.resolve("model.layers.0.self_attn.q_proj.weight", &[4096, 4096], 0, 1).unwrap();
        assert_eq!(solo.placement, Placement::Replicated);
        assert_eq!(solo.local_shape, vec![4096, 4096]);
    }

    #[test]
    fn moe_hybrid_expert_parallel_assigns_each_expert_to_exact_one_rank() {
        let plan = ShardSpecPlan::moe_hybrid(64, false, 128);
        for eid in 0..64 {
            let name = format!("model.layers.5.mlp.experts.{eid}.down_proj.weight");
            let owners: Vec<usize> = (0..4)
                .filter(|&r| plan.resolve(&name, &[4096, 1536], r, 4).unwrap().owned)
                .collect();
            assert_eq!(owners, vec![eid % 4], "expert {eid} must have exactly one owner");
        }
    }
}
