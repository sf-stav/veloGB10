//! S6 — the bundled DSpark / MTP drafter (`mtp.safetensors`) and exact speculative decoding at TP1.
//!
//!   * [`gpu`]    the drafter on the GPU (kernels/gpu_inkling2/a0_mtp.cuh) and the exact device accept;
//!   * [`cpu`]    the f32 CPU oracle of the drafter (op for op after the external SGLang reference);
//!   * [`policy`] the per-round draft count (exact look-ahead stop on the drafter's confidences) + CPU exactness tests;
//!   * [`spec`]   prefill / draft / verify / commit for one lane;
//!   * [`probe`]  `--probe-inkling2-mtp` (oracle, greedy identity, acceptance, speed);
//!   * [`serve`]  the serve worker with speculative rounds over the lanes (`--mtp auto|on`).

pub mod cpu;
pub mod gpu;
pub mod policy;
pub mod probe;
pub mod serve;
pub mod spec;
