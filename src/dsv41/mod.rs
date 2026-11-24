//! DeepSeek-V4.1-Flash-1017 (`model_type` = `deepseek_v41`): the dsv41 family. Plan:
//! `PLAN/DSV41_PLAN_2026-11-18.md`; the map of every piece to its leg is `src/dsv41/README.md`.
//!
//! The tree is RESERVED by leg I0 so later legs add files without editing this file: each
//! sub-module below exists (some as doc-only stubs) and is owned by the leg(s) its doc names.
//! Flash-Next and the other families are untouched: dsv41 code lives here and in
//! `kernels/gpu_dsv41*`, with one-line hooks in lib.rs, main.rs, build.rs, ptx_embed.rs,
//! startup.rs and the option table.

// ---- I0: skeleton and contracts ----
pub mod boot;
pub mod census;
pub mod cli;
pub mod config;
pub mod dump;
pub mod rankmap;
pub mod slice;
pub mod st;
pub mod state;

// ---- reserved for later legs (see README.md) ----
pub mod ced;     // CE: the exact CED prefill, chunking, prefix states, score mode
pub mod chat;    // ENC
pub mod cpu;     // OA, OM, OF, OD
pub mod engram;  // ENGa, ENGb, EIO
pub mod fwd;     // SL: the per-layer GPU forward (slice model) and its REFb gate
pub mod rope;    // SL: the ONE RoPE table builder (torch-on-CUDA semantics) for KC / KA / KI / SL / DS
pub mod gpu;     // kernel host side: I0 (abi), KC, KI, KA, KM, EIO, DS
pub mod load;    // LS
pub mod serve;   // SV
pub mod tp;      // T0, TP
pub mod vision;  // VIS1 (M4: ViT + 3x3 aligner, preproc, TP prefill splice, VL prefix cache)
pub mod weights; // I0 (contract), KE (exl3, plain), KN1 / KN2 (native)
