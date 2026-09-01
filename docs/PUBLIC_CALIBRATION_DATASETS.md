# Public, reproducible calibration corpora

`scripts/generate_public_v9_calibration_corpus.sh` and
`scripts/generate_public_v10_calibration_corpus.sh` build the v9 and v10 corpora without reading a
developer checkout, personal files, benchmark prompts or mutable dataset revisions. The complete
data path is implemented in Rust: `calib_sources` acquires the pinned API slices, reads
JSON/JSONL/Gzip/Parquet-Snappy, normalizes the conversations, deduplicates them and emits the
category pools; `calib_compose` renders the model chat template and assembles the token budget.
No Python runtime, `uv`, pandas or pyarrow is required. It is a calibration corpus, not training
data: it protects the relevant activation and Hessian regions during quantization but cannot
create a capability absent from the BF16 model.

## Public inputs

All raw inputs are public, ungated and checked before use:

| Purpose | Dataset / repository | Revision | License |
|---|---|---|---|
| English and long context | `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | ODC-BY |
| French | `FreedomIntelligence/alpaca-gpt4-french` | `79a2b0a3341c2bd4fcfa581eaf32f571d6eaa6cf` | Apache-2.0 |
| Multilingual | `CohereLabs/aya_dataset` | pinned API slice | Apache-2.0 |
| Math | `open-r1/OpenR1-Math-220k` | pinned API slice | Apache-2.0 |
| Public code | Go example, ShellCheck, TypeScript Website | pinned git commits | upstream licences |
| Tool reliability | `interstellarninja/toolace_sequential_tool_use_reasoning` | `d403e800de96bd7fec58902eddf431a485522a2f` | Apache-2.0 |
| Function schemas | `Johin/function-calling-dataset` | `ef3f5c4ce7cbf80b55f017fdb8695226cfad0976` | Apache-2.0 |

The prompt-injection seed is a versioned repository asset under
`assets/calibration/prompt_injection.jsonl`; it is not read from a machine-local corpus. Its
checksum appears in the generated source manifest and its variants are deterministic.

## Generate v9 from zero

```bash
cd ~/workspace/veloGB10

SEED=20260831 \
scripts/generate_public_v9_calibration_corpus.sh \
  "$HOME/models/Qwen3.8-27B" \
  "$HOME/models/calibration-sources/qwen38-calibration-v9-public-candidates.jsonl"
```

Set `EXCLUDE_JSONL=/path/to/held-out-evaluation.jsonl` to remove exact and near-duplicate
evaluation material before composition. Do not use an evaluation corpus as a selection reference.

The bootstrap checks every downloaded file by SHA-256. The Rust `parquet` reader handles the
8 MiB Snappy-compressed ToolACE snapshot directly. The generated `sources.manifest.json` records
the Rust generator identity, source identifiers, hashes, licences, category counts and the
deduplication result. Sampling uses a seeded ChaCha20 stream, so its algorithm is explicit rather
than depending on `rand`'s implementation-selected default RNG.

The shell files are thin orchestration wrappers around reproducible downloads, pinned Git
checkouts and the two Rust binaries. The actual corpus generation and transformation never invoke
Python. To inspect the Rust CLI directly:

```bash
cargo run --release --bin calib_sources -- --help
```

The exact 1,048,576-token MaCa prefix is 14% long multi-turn, 23% code, 19% multilingual, 15%
standard tools, 10% public agentic reliability trajectories, 9% public function-schema examples,
5% math and 5% prompt-injection defence. It increases coverage for recovery, conditional chains,
multi-turn state and exact arguments without copying evaluation prompts. Supply a held-out
evaluation JSONL through the existing `EXCLUDE_JSONL` mechanism whenever applicable.

Run `scripts/select_calibration_corpus.sh` and then the MaCa MR-GPTQ recipe in
`MACA_COLA_ACDM_MOE_CALIBRATION.md` using the candidate corpus it emits.

## Generate the reliability-focused v10 corpus

V10 keeps the same pinned public inputs but changes their normalization and token allocation to
cover the failure families seen in held-out agent/tool evaluations without copying evaluation
prompts. ToolACE rows are converted from XML-like transcripts to native `tool_calls` and matching
tool messages; incomplete, unmatched, or malformed trajectories are rejected. Johin argument
shapes are inferred across verified calls so the emitted schemas contain property types, required
fields, and `additionalProperties: false`.

```bash
cd ~/workspace/veloGB10

SEED=20260901 \
scripts/generate_public_v10_calibration_corpus.sh \
  "$HOME/models/Qwen3.8-27B" \
  "$HOME/models/calibration-sources/qwen38-calibration-v10-public-candidates.jsonl"
```

The exact 1,048,576-token MaCa prefix is:

- 14% long multi-turn;
- 20% code;
- 18% multilingual;
- 18% structured tools;
- 12% complete public agentic trajectories;
- 8% public function-schema examples;
- 5% verified math reasoning;
- 5% prompt-injection defence.

The structured-tool pool contains 16 balanced, generic reliability families: direct, sequential
and parallel calls; no-tool restraint; authorization and cancellation; malformed-response
fallback; asynchronous polling; stateful correction; exactly-once verification after an ambiguous
commit; accumulating constraints; schema restraint; precondition checking; information reveal;
and untrusted-output handling. Constructed v10 variants bypass only corpus-to-corpus
near-duplicate removal; exact duplicates and held-out benchmark exclusions still apply.

V10 passes `--trajectory-packing` to `calib_compose`. Conversation windows remain adjacent and a
partially consumed chunk resumes at its exact token offset, so later tool observations and final
answers are reachable. The manifest records `trajectory_packing`, per-window token counts, and
`late_window_tokens`; audit those fields before GPTQ. The low-level `calib_sources prepare`
command defaults to profile v9 for compatibility, while the v10 wrapper explicitly passes
`--profile v10`.

Calibration preserves activation/Hessian regions; it does not teach new behavior. When a failure
persists, compare the BF16 or W4A16 baseline with the quantized model before changing the corpus.
