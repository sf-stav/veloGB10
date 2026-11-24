# src/dsv41 — DeepSeek-V4.1-Flash-1017 (`deepseek_v41`)

Plan: `PLAN/DSV41_PLAN_2026-11-18.md` (Revision 1 first). Session rules: `PLAN/DSV41_SESSION_RULES.md`.
I0 reserved this tree so that every leg adds or edits only the files it owns. **Do not edit
another leg's file; do not edit `mod.rs` files you do not own** — every sub-module below already exists.

| Path | Owner | What |
|---|---|---|
| `mod.rs`, `README.md` | I0 | the tree; this map |
| `config.rs` | I0 | `Dsv41Config`: every `text_config` / `vision_config` / quantization field typed, `validate()` (43 ratios, KV / index / candidate wiring, Engram prime sums, shapes), `diff_dirs`, the reference ModelArgs cross-check |
| `census/` (`mod.rs`, `classify.rs`, `report.rs`, `synth.rs`, `tests.rs`) | I0 | the pack-generic tensor census: class / role / layer / format / shape / bytes of every tensor, equal to the index; formats from headers + `tensor_storage`; per-rank bytes; the markdown report |
| `st.rs` | I0 | safetensors header read / write (census, slices, dumps) |
| `rankmap.rs` | I0 (T0 refines) | TP2 / TP4 / TP8 -> ranks -> boxes -> Engram heads, experts, o_groups, vocab; the Engram bucket layout for byte accounting |
| `state.rs` | I0 | the state structs and byte accounting (main KV FP4+E4M3/16, index-K FP4+E8M0/32, SWA FP8+E8M0/32, frontiers, n-gram, DSpark carry) |
| `dump.rs` (+ `tools/dsv41_ref/dumpfmt.py`) | I0 | the tensor-dump format shared with the Python reference |
| `slice.rs` | I0 | `--dsv41-slice-pack`: mini-packs of chosen layers |
| `boot.rs` | I0 | the dsv41 rows of the startup configuration block |
| `cli.rs` | I0 (each leg fills its RESERVED arm) | the ONE mode hook main.rs calls |
| `weights/mod.rs` | I0 | the `Dsv41Weights` contract, `RowClass` (GEMV <= 2 / coop 3-16 / wide >= 17), `WeightPlan` |
| `weights/exl3/`, `weights/plain.rs` | KE | EXL3 trellis backend; plain BF16 / F16 / F32 linears (first-class, mixed packs) |
| `weights/native/dense.rs` | KN1 | FP8 32x32 ue8m0 dense |
| `weights/native/moe.rs` | KN2 | MXFP4 experts, FP8 activations |
| `gpu/mod.rs`, `gpu/abi.rs` | I0 | the `gpu_dsv41` module loader (build-id handshake, raw handles for PDL / smem opt-in), the ABI mirror and `--probe-dsv41-abi` |
| `gpu/kv.rs` | KC | compressor + cache writes |
| `gpu/indexer.rs` | KI | indexer, top-512, candidates |
| `gpu/attn.rs` | KA | sparse attention |
| `gpu/mixers.rs` | KM | mHC, norms, router, Engram gate |
| `gpu/engram.rs` | EIO | Engram row decode on the GPU |
| `gpu/dspark.rs` | DS | DSpark round glue |
| `cpu/attn.rs`, `cpu/mhc.rs`, `cpu/moe.rs`, `cpu/dspark.rs` | OA, OM, OF, OD | CPU f32 oracles |
| `engram/` | ENGa (`tokmap`, `layout`, `hash`), ENGb (packer), EIO (runtime) | Engram |
| `chat/` | ENC | encoding.py port + DSML parser |
| `load/` | LS (I0 wrote the boot-progress contract) | shard plan, shipping, upload, memory fit |
| `tp/` | T0, TP | the split design + the TP forward |
| `serve/` | SV | the engine loop on the shared server |
| `tools/dsv41_ref/` | REFb (I0: `dumpfmt.py`) | the Python reference |

Kernels: `kernels/gpu_dsv41.cu` + `kernels/gpu_dsv41/NN_*.cuh`, one band per leg (01 I0, 10 KE, 20 KN1,
30 KN2, 40 KC, 50 KI, 60 KA, 70 KM, 80 EIO, 90 DS). A band's head file (`10_trellis.cuh`, ...) is
included automatically once it exists; read `00_abi.cuh` first.

Verify widths (binding contract note, from KN1): row-class boundaries are backend-defined (KN1 serves rows 3..32 on its
cooperative split-K family); `RowClass::of` is only the nominal map. The invariant is what binds: every serving verify
width (<= 24 = 4 lanes x 6) produces column 0 bit-identical to a 1-row decode. Callers dispatch through the backend and
use the class it returns, never `RowClass::of`.

Adding things without touching shared files:
- a kernel: your band's files + your host file's `FNS`;
- a mode / probe: fill your RESERVED arm in `cli.rs` (the flag is already in `src/opts_table.rs`);
  a new flag goes into `src/opts_table.rs` (env `""`, the right `Vis`, re-bless the help snapshots);
- boot output: `boot_report::Stage` / `load::progress(Phase, ..)` / `boot_report::{warn,notice}`;
  diagnostics via `crate::dprintln!`; a direct print in a new file needs a logq baseline row.
