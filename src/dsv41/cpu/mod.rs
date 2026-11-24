//! DSV4.1 CPU f32 oracles (reference-grade, single-threaded is fine) — one file per oracle leg.
//! Every oracle reads the REFb dumps through `crate::dsv41::dump` and reports per-point rel-L2.
//!
//! | file | leg | what |
//! |---|---|---|
//! | `attn.rs` | OA | q / kv LoRA, MQA head_dim 512 + RoPE 64 (YaRN), compressor, indexer, candidates, SWA, o_groups |
//! | `mhc.rs` | OM | mHC single pass (hc 4, Sinkhorn 20), norms, router |
//! | `moe.rs` | OF | MoE (384 x top-6 + shared, SwiGLU limit 10) and the Engram module |
//! | `dspark.rs` | OD | the 3 DSpark stages, Markov head, confidence head |

pub mod attn;
pub mod dspark;
pub mod mhc;
pub mod moe;
