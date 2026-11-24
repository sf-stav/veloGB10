//! `inkling2` — the looped Inkling-2 text family (`model_type: "inkling2_loop"`, stock pack:
//! NVFP4 experts L0-19, MXFP8 experts L20-23, BF16 rest, bundled `mtp.safetensors` drafter) as its
//! own model family. Plan and stage ledger: `PLAN/INKLING2_MINI_PORT.md` (+ the owner notes).
//!
//! Discipline: code other families use is copied, never edited; shared library modules
//! (`fp4l::pack::StDir`, `server`, `tokenizer`, `tools`) are used read-only.
//!
//! Modules land stage by stage:
//!   * [`config`] S0 — the architecture, parsed from `config.json` (+ `mtp_config`).
//!   * [`pack`]   S0 — census of every tensor (model + drafter files), exact expert dequant,
//!                `--probe-inkling2-pack`.
//!   * [`cpu`]    S1 — the f32 CPU reference forward (op for op after the transformers modeling
//!                files), with prefill/decode state; `--probe-inkling2-cpu`.
//!   * [`model`]  S3 — the GPU model (TP1, stock pack), `--probe-inkling2-full`, the serve forward.
//!   * [`chat`]  T1 — the chat surface: request to prompt (reasoning effort, image splice points, tools), reply parser (token level).
//!   * [`serve`] T1 — the serve shell skeleton (stub forward), `--server --model-dir <Inkling-2 pack>`.
//!   * [`tok`]    S1 — chat-template / tokenizer id check vs transformers (`--probe-inkling2-tok`).
//!   * [`vision`] S5 — image input: preprocessing, the ViT + merger (CPU oracle, GPU), the prompt splice, `--probe-inkling2-vision`.
//!   * [`actq`]   Q1 — the opt-in expert ACTIVATION quantisation (`--inkling2-expert-act native`), `--probe-inkling2-actq`,
//!                `--probe-inkling2-logits`.
//!   * [`tp`]     S-INK2-TP2 — tensor parallelism (TP2 / TP4): the split, the TP kernels, the transport, head / node.
//!   * [`gpu`]    S2+ — the `gpu_inkling2` kernel module; K1: attention / convs / AttnRes / MessageGate /
//!                norms / projection GEMM, `--probe-inkling2-kernels`; K2: MoE router / experts / LM head, `--probe-inkling2-moe`.

pub mod config;
pub mod pack;
pub mod cpu;
pub mod tok;
pub mod gpu;
pub mod model;
pub mod mtp;
pub mod chat;
pub mod serve;
pub mod vision;
pub mod actq;
pub mod tp;
