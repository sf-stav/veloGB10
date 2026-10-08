# Credits

We would like to thank the following people for their contributions to this project. The per-release credits are in [CHANGELOG.md](CHANGELOG.md); upstream projects and models are acknowledged in the README.

## Code

- [@vcruz305](https://github.com/vcruz305) — the EXL3 implementation and recipe the engine's EXL3 path grew from (see the README acknowledgements).
- [@kedric](https://github.com/kedric) — pull request [#4](https://github.com/sf-stav/veloGB10/pull/4) (Qwen3.8-Flash-Next work, NVFP4 W4A4 prefill kernels) and the calibration recipes (layerwise MaCa, variable-length IGS).
- [@Morxi](https://github.com/Morxi) — pull request [#5](https://github.com/sf-stav/veloGB10/pull/5) (RDMA address discovery on point-to-point prefixes).
- [@seanlinmt](https://github.com/seanlinmt) — pull requests [#15](https://github.com/sf-stav/veloGB10/pull/15) (TP network bring-up: RoCEv2 GID auto-detect, explicit CUDA device, clearer allocation errors), [#16](https://github.com/sf-stav/veloGB10/pull/16) (vision tower error handling and recovery) and [#17](https://github.com/sf-stav/veloGB10/pull/17) (TP image payload ordering, PLE headroom). Ported; shipping in the next release.
- [@ccgauvin94](https://github.com/ccgauvin94) — pull requests [#21](https://github.com/sf-stav/veloGB10/pull/21) (device-memory fix for windows past the trained length) and [#22](https://github.com/sf-stav/veloGB10/pull/22) (YaRN). Porting planned.
- [@adriangrassi](https://github.com/adriangrassi) — pull request [#23](https://github.com/sf-stav/veloGB10/pull/23) (`--prefill-interleave`). Porting planned.

## Reports, requests and benchmarks

- [@JashicTM](https://github.com/JashicTM) — [#6](https://github.com/sf-stav/veloGB10/issues/6) (tool-call history rendering) and [#10](https://github.com/sf-stav/veloGB10/issues/10) (the rare TP=2 pre-verify abort, with watchdog dumps and a precise analysis; fixed in v0.7.3).
- [@liorm0505](https://github.com/liorm0505) — [#8](https://github.com/sf-stav/veloGB10/issues/8) (server surface: context length, cached tokens, model name; fixed in v0.7.3) and [#20](https://github.com/sf-stav/veloGB10/issues/20) (vision tower lock poisoned by a cuBLAS error; fixed on the development line).
- [@liorzivsensors-cmd](https://github.com/liorzivsensors-cmd) — [#7](https://github.com/sf-stav/veloGB10/issues/7) (the first filing of the server-surface report).
- [@MushroomMan321](https://github.com/MushroomMan321) — [#9](https://github.com/sf-stav/veloGB10/issues/9) (packs from exllamav3 1.5.x with the sharded n-gram sidecar, with a working analysis and workaround; fixed in v0.7.3).
- [@npw1980](https://github.com/npw1980) — [#2](https://github.com/sf-stav/veloGB10/issues/2) (v0.5.0 boot crash on non-27B packs; fixed in v0.5.1).
- [@herbertp](https://github.com/herbertp) — [#1](https://github.com/sf-stav/veloGB10/issues/1) (out-of-memory on two nodes with Hy3).
- [@hatemismail](https://github.com/hatemismail) — [#18](https://github.com/sf-stav/veloGB10/issues/18) (repeatable reasoning-budget exhaustion, with a complete reproducer; under investigation).
- [@ccgauvin94](https://github.com/ccgauvin94) — [#19](https://github.com/sf-stav/veloGB10/issues/19) (YaRN support) and the confirmation of [#10](https://github.com/sf-stav/veloGB10/issues/10).

and the community members on the NVIDIA developer forum who posted logs, benchmarks and bug reports.
