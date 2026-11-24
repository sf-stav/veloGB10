# Credits

We would like to thank the following people for their contributions to this project. The per-release credits are in [CHANGELOG.md](CHANGELOG.md); upstream projects and models are acknowledged in the README.

## Code

- [@vcruz305](https://github.com/vcruz305) — the EXL3 implementation and recipe the engine's EXL3 path grew from (see the README acknowledgements).
- [@kedric](https://github.com/kedric) — pull request [#4](https://github.com/sf-stav/veloGB10/pull/4) (Qwen3.8-Flash-Next work, NVFP4 W4A4 prefill kernels) and the calibration recipes (layerwise MaCa, variable-length IGS).
- [@Morxi](https://github.com/Morxi) — pull request [#5](https://github.com/sf-stav/veloGB10/pull/5) (RDMA address discovery on point-to-point prefixes).
- [@seanlinmt](https://github.com/seanlinmt) — pull requests [#15](https://github.com/sf-stav/veloGB10/pull/15) (TP network bring-up: RoCEv2 GID auto-detect, explicit CUDA device, clearer allocation errors), [#16](https://github.com/sf-stav/veloGB10/pull/16) (vision tower error handling and recovery) and [#17](https://github.com/sf-stav/veloGB10/pull/17) (TP image payload ordering, PLE headroom). Ported; shipped in v0.7.4.
- [@ccgauvin94](https://github.com/ccgauvin94) — pull request [#21](https://github.com/sf-stav/veloGB10/pull/21) (device-memory fix for windows past the trained length; ported, shipped in v0.7.4) and pull request [#22](https://github.com/sf-stav/veloGB10/pull/22) (YaRN; porting planned).
- [@adriangrassi](https://github.com/adriangrassi) — pull request [#23](https://github.com/sf-stav/veloGB10/pull/23) (`--prefill-interleave`). Ported; shipped in v0.7.4.
- [@andersc](https://github.com/andersc) — pull request [#29](https://github.com/sf-stav/veloGB10/pull/29) (top_k client compatibility: negative values accepted as "no top-k truncation" on both endpoints, CPU-sampler edge cases fixed). Ported; shipped in v0.7.8.
- [@rusty-dev](https://github.com/rusty-dev) — pull request [#42](https://github.com/sf-stav/veloGB10/pull/42) (VISCACHE: content-keyed vision encoder cache across multi-turn image conversations). Ported; shipped in v0.8.0.
- [@AbsolomBane](https://github.com/AbsolomBane) — [#43](https://github.com/sf-stav/veloGB10/issues/43) (CUDA VMM shared KV page pool across lanes, `--kv-pool` / `--fit-floor-gb`) and [#44](https://github.com/sf-stav/veloGB10/issues/44) (TP fix: failed spill capture still votes in the agree). Ported; shipped in v0.8.0.
- [@jetnet](https://github.com/jetnet) — pull request [#47](https://github.com/sf-stav/veloGB10/pull/47) (gpu-wait covers DFlash2 / DSpark draft rounds). Ported; shipped in v0.8.0.
- [@MiaAI-Lab](https://github.com/MiaAI-Lab) — DeepSeek-V4.1-Flash-1017 model & reference work (supported end-to-end in v0.8.0).

## Reports, requests and benchmarks

- [@JashicTM](https://github.com/JashicTM) — [#6](https://github.com/sf-stav/veloGB10/issues/6) (tool-call history rendering) and [#10](https://github.com/sf-stav/veloGB10/issues/10) (the rare TP=2 pre-verify abort, with watchdog dumps and a precise analysis; fixed in v0.7.3).
- [@liorm0505](https://github.com/liorm0505) — [#8](https://github.com/sf-stav/veloGB10/issues/8) (server surface: context length, cached tokens, model name; fixed in v0.7.3), [#20](https://github.com/sf-stav/veloGB10/issues/20) (vision tower lock poisoned by a cuBLAS error; fixed in v0.7.4) and [#24](https://github.com/sf-stav/veloGB10/issues/24) (KV pool occupancy gauges on `/metrics`; shipped in v0.7.4).
- [@liorzivsensors-cmd](https://github.com/liorzivsensors-cmd) — [#7](https://github.com/sf-stav/veloGB10/issues/7) (the first filing of the server-surface report).
- [@MushroomMan321](https://github.com/MushroomMan321) — [#9](https://github.com/sf-stav/veloGB10/issues/9) (packs from exllamav3 1.5.x with the sharded n-gram sidecar, with a working analysis and workaround; fixed in v0.7.3).
- [@npw1980](https://github.com/npw1980) — [#2](https://github.com/sf-stav/veloGB10/issues/2) (v0.5.0 boot crash on non-27B packs; fixed in v0.5.1).
- [@herbertp](https://github.com/herbertp) — [#1](https://github.com/sf-stav/veloGB10/issues/1) (out-of-memory on two nodes with Hy3).
- [@hatemismail](https://github.com/hatemismail) — [#18](https://github.com/sf-stav/veloGB10/issues/18) (repeatable reasoning-budget exhaustion, with a complete reproducer; investigated for v0.7.4: no engine defect found, the reasoning length depends on how the tool definitions are serialised in the prompt).
- [@ccgauvin94](https://github.com/ccgauvin94) — [#19](https://github.com/sf-stav/veloGB10/issues/19) (YaRN support) and the confirmation of [#10](https://github.com/sf-stav/veloGB10/issues/10).
- [@ablake4427](https://github.com/ablake4427) — [#45](https://github.com/sf-stav/veloGB10/issues/45) (TP=2 pairwise rendezvous before back-to-back agrees, fixing `--spec-arm-meter v2` agree stall) and [#46](https://github.com/sf-stav/veloGB10/issues/46) (least-recently-released free lane selection on unscoped admission fallback). Fixed in v0.8.0.

- **bddh** (NVIDIA developer forum) — the payloads and the notes on how tool-call arguments that quote tool calls are read, which drove the parser work of v0.7.4.
- **TimoGoetze** (NVIDIA developer forum) — the llama-benchy tables that showed the first streamed chunk arriving long before any output (fixed in v0.7.4) and the concurrency numbers behind the lane-packing work.
- **cdzinpin** (NVIDIA developer forum) — the server logs behind the speculation re-arm fix (v0.7.4).
- **p1140** and **emX0r** (NVIDIA developer forum) — reports of an agent stopping where a tool call was expected (the `[gen]` stop fields of v0.7.4 make such a stop readable) and of a memory spike (under investigation); no fix is claimed.

and the community members on the NVIDIA developer forum who posted logs, benchmarks and bug reports.
