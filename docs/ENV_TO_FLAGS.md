# Environment variables -> command-line flags (v0.7.0)

Since v0.7.0 **every engine option is a command-line flag; the engine reads no
environment variables.** The binary refuses to start when one of the variables below is set, naming
its flag. The registry that defines every option is `src/opts_table.rs` (`src/opts.rs` explains the
types); `--help` lists the options, `--help-diag` the diagnostics / test drills, `--print-config` every
option's resolved value for a given command line.

Value syntax: `--x` / `--x on|off` / `--x=V` / `--x V`. A **flag** is a presence switch (`--x` = on,
`--x off` = unset). An **on/off** option stores `1` / `0` (the old `=1` / `=0`). Under TP every option
is set on the HEAD only: the head ships its resolved options to the node (TpConfig v23 `opts`); the
node's own command line carries only the per-box options (scope `local`).

| env var (removed) | flag | type | default | scope | what |
|---|---|---|---|---|---|
| `GB10_ACCEPT_DEBUG` | `--accept-debug` | flag | off | spmd | NVFP4 MTP: print per-step acceptance / draft details |
| `GB10_ATTN_E_DEBUG` | `--attn-e-debug` | flag | off | spmd | NVFP4 attention-E path debug prints |
| `GB10_ATTN_FLASH` | `--attn-flash` | on/off | on | spmd | EXL3 flash attention path (off = the pre-flash kernels) |
| `GB10_ATTN_P3` | `--attn-p3` | flag | off | spmd | NVFP4 P3 attention variant (opt-in) |
| `GB10_A_CTRL` | `--a-ctrl` | one of new / fixed128 / old | new | spmd | WP-A escape: lane control (fixed128 / old = the pre-WP03 control) |
| `GB10_A_OFF` | `--a-off` | flag | off | spmd | WP-A escape: every WP-A lane fix off (the pre-WP03 behaviour) |
| `GB10_A_REMCLAMP` | `--a-remclamp` | on/off | on | spmd | WP-A escape: remaining-budget clamp (off = unclamped) |
| `GB10_A_SEAM` | `--a-seam` | one of new / headprime / old | new | spmd | WP-A escape: prime seam (headprime / old) |
| `GB10_A_TAPS` | `--a-taps` | one of lane / shared | lane | spmd | WP-A escape: shared = one tap buffer for every lane (the pre-WP03 clobber) |
| `GB10_A_XCHECK` | `--a-xcheck` | flag | off | spmd | WP03 cross-lane tap bit-compare every round (syncs) |
| `GB10_BENCH_LOGITS_HASH` | `--bench-logits-hash` | flag | off | spmd | bench: print a logits hash per step |
| `GB10_BINV_SECTION` | `--binv-section` | one of wp18 / moe_pf / pf_hc / gdn_tc | all | spmd | EXL3 --probe-binv: run only this section |
| `GB10_BISECT_LEN` | `--bisect-len` | int | unset | spmd | DSV4 TP bench: pad/truncate the prompt to exactly N tokens |
| `GB10_BMTP_DIAG` | `--bmtp-diag` | one of plain / noprime | unset | spmd | EXL3 bench-mtp diagnostic mode |
| `GB10_CAP_DEBUG` | `--cap-debug` | flag | off | spmd | graph-capture debug prints |
| `GB10_DBG_MTP_ONLY` | `--dbg-mtp-only` | text | unset | spmd | NVFP4 MTP debug: restrict the debug dump to this sub-stage |
| `GB10_DEBUG_HASH` | `--debug-hash` | flag | off | spmd | NVFP4 per-step debug hashes |
| `GB10_DEVICE_LOOP` | `--device-loop` | one of on / off / 1 / 0 / true / false | off | spmd | device-resident token loop (NVFP4; under TP rides TpConfig) |
| `GB10_DF2_CAPTURE` | set by `--df2-capture on` | internal | off | spmd | DFlash2 trunk tap capture (set by --df2-capture) |
| `GB10_DF2_CARRY_LEN_ONLY` | `--df2-carry-len-only` | flag | off | spmd | DF2 carry diagnostic: carry the ring length only |
| `GB10_DF2_DET_DUAL` | `--df2-det-dual` | flag | off | spmd | DF2 determinism probe: dual run |
| `GB10_DF2_DET_ONLY` | `--df2-det-only` | flag | off | spmd | DF2 determinism probe: determinism check only |
| `GB10_DF2_DET_POST` | `--df2-det-post` | flag | off | spmd | DF2 determinism probe: post-round check |
| `GB10_DF2_GATE_DUMP` | `--df2-gate-dump` | flag | off | spmd | DF2 gate dump |
| `GB10_DF2_MIRROR_THREADS` | `--df2-mirror-threads` | int | auto | spmd | DFlash2 mirror worker threads |
| `GB10_DF2_QHEAD` | `--df2-qhead` | flag | off | spmd | DFlash2: plain-q head borrow (bf16 borrow only) |
| `GB10_DF2_STEP_LOG` | `--df2-step-log` | flag | off | spmd | DFlash2 per-step log |
| `GB10_DF2_TP` | set by `--spec-source dflash2*/dspark` | internal | off | spmd | keep the full lm_head for the DF2/DSpark round under TP (derived from --spec-source) |
| `GB10_DFLASH_ATTN_EAGER` | `--dflash-attn-eager` | flag | off | spmd | DFlash v1: eager (untiled) drafter attention |
| `GB10_DFLASH_DEBUG` | `--dflash-debug` | flag | off | spmd | DFlash v1 debug prints |
| `GB10_DFLASH_STEP_LOG` | `--dflash-step-log` | flag | off | spmd | DFlash v1 per-step log |
| `GB10_DFLASH_TAPDUMP` | `--dflash-tapdump` | path | unset | spmd | DFlash v1: dump the trunk taps to this path |
| `GB10_DHEADP_CAP` | `--dheadp-cap` | int | 256 | spmd | DHEADP expansion cap (tunable dheadp.cap override) |
| `GB10_DHEADP_OFF` | `--dheadp-off` | on/off | off | spmd | DHEADP escape (tunable dheadp.off override) |
| `GB10_DHEAD_BITS` | `--dhead-bits` | one of 1 / 2 / 4 | 2 | spmd | DHEAD screen bits |
| `GB10_DHEAD_DELTA` | `--dhead-delta` | number | 4.0 | spmd | DHEAD rescore margin |
| `GB10_DHEAD_OFF` | `--dhead-off` | flag | off | spmd | DHEAD escape: the full draft head |
| `GB10_DHEAD_T` | `--dhead-t` | int | 16 | spmd | DHEAD candidate count |
| `GB10_DHEAD_XCHECK` | `--dhead-xcheck` | flag | off | spmd | DHEAD eager cross-check vs the full head |
| `GB10_DISPATCH_LOG` | `--dispatch-log` | one of 1 / count / assert | off | spmd | dispatch assertion posture (count / assert) |
| `GB10_DRAFT_DIR` | set by `--draft-dir` | internal | unset | spmd | drafter artifact dir (set by --draft-dir) |
| `GB10_DSPARK_ATTN` | `--dspark-attn` | text | default | spmd | DSpark round attention selector (diagnostic) |
| `GB10_DSPARK_BISECT` | `--dspark-bisect` | flag | off | spmd | DSpark bisect mode |
| `GB10_DSPARK_BLOCK_POS` | `--dspark-block-pos` | int | 0 | spmd | DSpark block position override |
| `GB10_DSPARK_CHAIN_A` | `--dspark-chain-a` | flag | off | spmd | DSpark legacy chain A |
| `GB10_DSPARK_D0_PLAIN` | `--dspark-d0-plain` | flag | off | spmd | DSpark legacy plain d0 |
| `GB10_DSPARK_DEBUG` | `--dspark-debug` | flag | off | spmd | DSpark debug prints |
| `GB10_DSPARK_DEBUG_RING` | `--dspark-debug-ring` | flag | off | spmd | DSpark ring debug prints |
| `GB10_DSPARK_DUMP_STEP0` | `--dspark-dump-step0` | path | unset | spmd | DSpark: dump step 0 to this dir |
| `GB10_DSPARK_DUMP_STEPS` | `--dspark-dump-steps` | int | unset | spmd | DSpark: number of steps to dump |
| `GB10_DSPARK_FP8_LOGITS` | `--dspark-fp8-logits` | flag | off | spmd | DSpark fp8 draft LM head (also --dspark-fp8-head on the TP head) |
| `GB10_DSPARK_GRAPH` | `--dspark-graph` | flag | off | spmd | DSpark graphed round |
| `GB10_DSPARK_NOMSCALE` | `--dspark-nomscale` | flag | off | spmd | DSpark: no mscale (legacy) |
| `GB10_DSPARK_NORM_G1` | `--dspark-norm-g1` | flag | off | spmd | DSpark: legacy norm g1 |
| `GB10_DSPARK_PHASE_MS` | `--dspark-phase-ms` | flag | off | spmd | DSpark per-phase ms timing (syncs) |
| `GB10_DSPARK_RING_BLIND` | `--dspark-ring-blind` | flag | off | spmd | DSpark ring-blind diagnostic |
| `GB10_DSPARK_ROPE_IL` | `--dspark-rope-il` | flag | off | spmd | DSpark interleaved rope (legacy) |
| `GB10_DSPARK_ROUND_BISECT` | `--dspark-round-bisect` | int | unset | spmd | DSpark round layer bisect |
| `GB10_DSPARK_ROUND_TRACE` | `--dspark-round-trace` | flag | off | spmd | DSpark round trace |
| `GB10_DSPARK_STEP_LOG` | `--dspark-step-log` | flag | off | spmd | DSpark per-step log |
| `GB10_DSPARK_TAP_REV` | `--dspark-tap-rev` | flag | off | spmd | DSpark reversed tap order |
| `GB10_DSPARK_TAP_SHIFT` | `--dspark-tap-shift` | int | 0 | spmd | DSpark tap shift |
| `GB10_DUMP_PFHASH` | `--dump-pfhash` | flag | off | spmd | NVFP4 prefill hash dumps |
| `GB10_E8_NO_SHARD` | `--e8-no-shard` | flag | off | spmd | E8 escape: keep the shared expert replicated under TP |
| `GB10_E9_NO_FOLD` | `--e9-no-fold` | flag | off | spmd | E9 escape: no programmatic dependent-launch overlap |
| `GB10_EXACT_GEMM` | `--exact-gemm` | flag | off | spmd | select the locked bit-exact GEMM kernels (tolerance-class fast paths are the default) |
| `GB10_EXL3_A1B3` | `--exl3-a1b3` | on/off | on | spmd | A-once 3-bit expert entry (tunable moe.a1b3 override) |
| `GB10_EXL3_CHAIN_PAIR` | `--exl3-chain-pair` | on/off | on | spmd | paired same-input chains (tunable override) |
| `GB10_EXL3_CHAIN_PAIR_CHECK` | `--exl3-chain-pair-check` | flag | off | spmd | EXL3 chain-pair eager cross-check (graphs off) |
| `GB10_EXL3_CONV_SERIAL` | `--exl3-conv-serial` | flag | off | spmd | prefill conv serial (tunable pf.conv_par=0) |
| `GB10_EXL3_DDS_DEPTH_BINS` | `--exl3-dds-depth-bins` | one of on / 1 / off / 0 | auto | spmd | WP23 depth bins (default: on iff depth > the legacy max) |
| `GB10_EXL3_DDS_GUARD` | `--exl3-dds-guard` | one of on / 1 / off / 0 | auto | spmd | WP23 cost guard (default: on iff depth > the legacy max) |
| `GB10_EXL3_DENSE_V3` | `--exl3-dense-v3` | on/off | on | spmd | WP13 dense attention v3 (tunable override) |
| `GB10_EXL3_DENSE_XCHECK` | `--exl3-dense-xcheck` | flag | off | spmd | WP13 dense attention eager cross-check |
| `GB10_EXL3_DRAFT_ATTN` | `--exl3-draft-attn` | one of legacy | new | spmd | draft attention: legacy = the pre-WP path |
| `GB10_EXL3_DRAFT_GRAPH` | `--exl3-draft-graph` | on/off | on | spmd | EXL3 draft graphs (tunable override) |
| `GB10_EXL3_DUMP8` | `--exl3-dump8` | on/off | on | spmd | EXL3 8-bit trellis dump path at load |
| `GB10_EXL3_ESEL_HIST` | `--exl3-esel-hist` | flag | off | spmd | EXL3 expert-selection histogram |
| `GB10_EXL3_GATHER_XCHECK` | `--exl3-gather-xcheck` | flag | off | spmd | WP14 QSA gather eager cross-check |
| `GB10_EXL3_GDN_RING` | `--exl3-gdn-ring` | on/off | on | spmd | GDN state ring (off = the pre-ring path) |
| `GB10_EXL3_GDN_RING_CHECK` | `--exl3-gdn-ring-check` | flag | off | spmd | GDN ring eager check |
| `GB10_EXL3_GDN_TC_MIN` | `--exl3-gdn-tc-min` | int | built-in | spmd | GDN tensor-core chunk min rows |
| `GB10_EXL3_GDN_TC_PREC` | `--exl3-gdn-tc-prec` | text | default | spmd | GDN tensor-core precision variant |
| `GB10_EXL3_GDN_XCHECK` | `--exl3-gdn-xcheck` | flag | off | spmd | GDN tensor-core eager cross-check |
| `GB10_EXL3_HEAD_FILL` | `--exl3-head-fill` | on/off | on | spmd | EXL3 head fill |
| `GB10_EXL3_HEAD_QSA` | `--exl3-head-qsa` | on/off | on | spmd | EXL3 QSA on the draft head (off = dense) |
| `GB10_EXL3_MOE_DEVROUTE` | `--exl3-moe-devroute` | on/off | on | spmd | EXL3 device-side MoE routing |
| `GB10_EXL3_MOE_EPI_CHECK` | `--exl3-moe-epi-check` | flag | off | spmd | EXL3 MoE epilogue check |
| `GB10_EXL3_MOE_GROUPED` | `--exl3-moe-grouped` | on/off | on | spmd | prefill grouped expert launches (tunable override) |
| `GB10_EXL3_MOE_RECON_MIN` | `--exl3-moe-recon-min` | int | 0 | spmd | MoE reconstruct min rows |
| `GB10_EXL3_MTP_HEAD_N` | `--exl3-mtp-head-n` | int | auto | spmd | EXL3 MTP head count override |
| `GB10_EXL3_MTP_K` | `--exl3-mtp-k` | int | auto | spmd | EXL3 MTP depth k override |
| `GB10_EXL3_NO_GRAPH` | `--exl3-no-graph` | flag | off | spmd | EXL3: no CUDA graphs (diagnostic escape) |
| `GB10_EXL3_NO_MTP` | `--exl3-no-mtp` | flag | off | spmd | EXL3: no MTP (diagnostics hatch ONLY, AGENTS 1b) |
| `GB10_EXL3_NO_PLE` | `--exl3-no-ple` | flag | off | spmd | EXL3: no PLE (diagnostic) |
| `GB10_EXL3_NO_PREFILL_WARMUP` | `--exl3-no-prefill-warmup` | flag | off | spmd | skip the serve-boot prefill warmup |
| `GB10_EXL3_PDL` | `--exl3-pdl` | on/off | off | spmd | PDL per graph family (tunable override) |
| `GB10_EXL3_PEN_DRAFT` | `--exl3-pen-draft` | on/off | on | spmd | penalties applied to drafts |
| `GB10_EXL3_POISON` | `--exl3-poison` | flag | off | spmd | EXL3 scratch poisoning |
| `GB10_EXL3_PREFIX` | set by `--prefix-cache on` | internal | off | spmd | EXL3 prefix-cache snapshots (set by --prefix-cache) |
| `GB10_EXL3_QSA_SPLITS_DEC` | `--exl3-qsa-splits-dec` | int | 32 | spmd | QSA decode splits (N-class tunable override) |
| `GB10_EXL3_RECON_MIN` | `--exl3-recon-min` | int | built-in | spmd | dense reconstruct min rows |
| `GB10_EXL3_REPRIME` | `--exl3-reprime` | one of 0 / full / on | on | spmd | device re-prime mode (0 = off, full) |
| `GB10_EXL3_REPRIME_EAGER` | `--exl3-reprime-eager` | flag | off | spmd | eager device re-prime |
| `GB10_EXL3_REPRIME_GRAPH` | `--exl3-reprime-graph` | on/off | on | spmd | graphed device re-prime (tunable override) |
| `GB10_EXL3_ROUND_PROF` | `--exl3-round-prof` | flag | off | spmd | per-round phase profiler |
| `GB10_EXL3_ROUTER_FUSED` | `--exl3-router-fused` | on/off | on | spmd | fused router (tunable override) |
| `GB10_EXL3_ROUTER_KS` | `--exl3-router-ks` | text | built-in | spmd | router split-K list |
| `GB10_EXL3_ROUTER_ROWS` | `--exl3-router-rows` | on/off | on | spmd | prefill router rows32 twin (tunable override) |
| `GB10_EXL3_ROUTER_XCHECK` | `--exl3-router-xcheck` | flag | off | spmd | fused router eager cross-check |
| `GB10_EXL3_ROUTE_CHECK` | `--exl3-route-check` | flag | off | spmd | EXL3 routing check (graphs off) |
| `GB10_EXL3_SCORE_MR` | `--exl3-score-mr` | one of 0 / off / 1 / on / generic | on | spmd | WP04 bucket-free scorer (0 = bucketed; generic body) |
| `GB10_EXL3_SCORE_MR_CTAS` | `--exl3-score-mr-ctas` | int | 4 | spmd | WP04 scorer CTAs/SM (tunable override) |
| `GB10_EXL3_SCORE_XCHECK` | `--exl3-score-xcheck` | flag | off | spmd | WP04 scorer eager cross-check |
| `GB10_EXL3_SEL_GATHER` | `--exl3-sel-gather` | text | -1 (auto) | spmd | WP04 gather variant pin (tunable override) |
| `GB10_EXL3_SEL_VEC` | `--exl3-sel-vec` | on/off | on | spmd | legacy gather float4 twin (tunable override) |
| `GB10_EXL3_SEL_XCHECK` | `--exl3-sel-xcheck` | flag | off | spmd | WP04 gather eager cross-check |
| `GB10_EXL3_TOPK_V2` | `--exl3-topk-v2` | on/off | on | spmd | WP04 top-k asc2 (tunable override) |
| `GB10_EXL3_TOPK_XCHECK` | `--exl3-topk-xcheck` | flag | off | spmd | WP04 top-k eager cross-check |
| `GB10_EXL3_TP_FASTCTL` | `--exl3-tp-fastctl` | on/off | on | spmd | EXL3 TP fast control plane |
| `GB10_EXL3_TP_IDENT` | `--exl3-tp-ident` | flag | off | spmd | EXL3 TP served ident digests across ranks every lockstep point |
| `GB10_EXL3_TP_PREVERIFY` | `--exl3-tp-preverify` | on/off | on | spmd | EXL3 TP pre-verify agree |
| `GB10_EXL3_WIDE_MT` | `--exl3-wide-mt` | one of 1 / 2 / 4 / 8 | 0 (the MT rule; 4 for dense) | spmd | wide-M trellis GEMM m-tiles per CTA (engine override; also the --bench-exl3-wide MT) |
| `GB10_F9_XHASH` | `--f9-xhash` | int or bare | off | spmd | NVFP4 F9 cross-hash dumps (1/2/3) |
| `GB10_FA_PREFILL` | `--fa-prefill` | on/off | on | spmd | NVFP4 tensor-core flash-attention prefill |
| `GB10_FOLD_XCHAIN_DUMP` | `--fold-xchain-dump` | path | unset | spmd | NVFP4 fold cross-chain dump path |
| `GB10_FP8_PREFILL` | `--fp8-prefill` | on/off | on | spmd | native e4m3 W8A8 prefill lane |
| `GB10_FUSE_RESIDUAL` | `--fuse-residual` | flag | off | spmd | FFN-epilogue residual fusion (E14) |
| `GB10_GDN_CHUNK` | `--gdn-chunk` | flag | off | spmd | NVFP4 chunkwise GDN prefill (v1) |
| `GB10_GDN_CHUNK2` | `--gdn-chunk2` | on/off | on | spmd | NVFP4 chunkwise GDN prefill v2 |
| `GB10_GDN_OLDSTAGE` | `--gdn-oldstage` | flag | off | spmd | NVFP4 GDN old staging kernel |
| `GB10_GDN_ROLLBACK_D2D` | `--gdn-rollback-d2d` | flag | off | spmd | NVFP4 GDN rollback via D2D copies (escape) |
| `GB10_GDN_SPLIT` | `--gdn-split` | flag | off | spmd | NVFP4 split chunked GDN prefill (diagnostic variant) |
| `GB10_GDN_TIME` | `--gdn-time` | flag | off | spmd | NVFP4 GDN per-layer timing (syncs) |
| `GB10_GDN_XCHECK` | `--gdn-xcheck` | flag | off | spmd | NVFP4 GDN eager cross-check |
| `GB10_GEMM_SPLITK` | `--splitk-gemm` | int or bare | off | spmd | NVFP4 serving-GEMM split-K (bare = per-shape auto; 0 = off; N>=2 forces N, single-node only) |
| `GB10_GRAPH` | `--graph` | flag | off | spmd | DSV4 decode graphs |
| `GB10_GRAPH_DEBUG` | `--graph-debug` | flag | off | spmd | DSV4 graph debug prints |
| `GB10_GRAPH_NOUPDATES` | `--graph-noupdates` | flag | off | spmd | DSV4 graph: skip node updates |
| `GB10_GRAPH_UPLOAD` | `--graph-upload` | flag | off | spmd | DSV4 graph upload |
| `GB10_HC_FP16` | `--hc-fp16` | flag | off | spmd | EXL3 hyper-connection mixers in bit-exact fp16 (int8 is the default) |
| `GB10_HC_FUSE` | `--hc-fuse` | on/off | on | spmd | EXL3 hc fuse |
| `GB10_HC_INT8` | `--hc-int8` | on/off | on | spmd | EXL3 int8 hc mixers (off = fp16; prefer --hc-fp16) |
| `GB10_HC_INT8_CHECK` | `--hc-int8-check` | flag | off | spmd | EXL3 hc int8 check (graphs off) |
| `GB10_HC_MIX4` | `--hc-mix4` | on/off | on | spmd | shared inject reduction tree (tunable override) |
| `GB10_HC_RB` | `--hc-rb` | on/off | on | spmd | row-batched int8 hc mixer (tunable override) |
| `GB10_HC_STEP_CHECK` | `--hc-step-check` | flag | off | spmd | EXL3 hc step eager check |
| `GB10_INJECT_PANIC_STEPS` | `--inject-panic-steps` | int | unset | spmd | test drill: inject a scheduler panic at decode step N |
| `GB10_KV_CACHE` | set by `--kv-cache f32/f16/fp8/q8` | internal | default | spmd | EXL3 KV cache format (set by --kv-cache) |
| `GB10_KV_K8V4` | set by `--kv-cache k8v4` | internal | off | spmd | NVFP4 int8-K + q4-V KV cache (set by --kv-cache k8v4) |
| `GB10_KV_K8V8` | set by `--kv-cache k8v8` | internal | off | spmd | NVFP4 int8 K+V KV cache (set by --kv-cache k8v8) |
| `GB10_KV_MIRROR_BUDGET` | `--kv-mirror-budget` | int | 32768 | spmd | packed-KV mirror budget (positions) |
| `GB10_KV_QUANT` | set by `--kv-cache q4` | internal | off | spmd | NVFP4 4-bit KV cache (set by --kv-cache q4) |
| `GB10_KV_TQ` | set by `--kv-cache tq/tq3` | internal | off | spmd | NVFP4 TurboQuant KV (1 = b2 K, 3 = b3 K; set by --kv-cache tq/tq3) |
| `GB10_LAYER_CSUM` | `--layer-csum` | flag | off | spmd | NVFP4 per-layer checksums |
| `GB10_LAYER_FULLHASH` | `--layer-fullhash` | flag | off | spmd | NVFP4 per-layer full hashes |
| `GB10_LMH_HAD` | `--lmh-had` | on/off | on | spmd | W3/LMH fused output Hadamard (tunable override) |
| `GB10_LMH_NST` | `--lmh-nst` | one of 4 / 6 / 8 | 6 | spmd | W3/LMH cp.async ring depth (tunable override) |
| `GB10_LMH_OFF` | `--lmh-off` | on/off | off | spmd | W3/LMH escape (tunable override) |
| `GB10_LMH_XCHECK` | `--lmh-xcheck` | flag | off | spmd | W3/LMH eager cross-check |
| `GB10_LOAD_FORCE` | `--load-force` | one of 1 / unsafe | off | spmd | bypass the load memory guard (unsafe = also on qwen4_exp) |
| `GB10_LOAD_PIPE_CAP_GB` | `--load-pipe-cap-gb` | int | built-in | spmd | load pipeline cap (GB) |
| `GB10_LOAD_SHARD_TRACE` | `--load-shard-trace` | flag | off | spmd | load: shard trace |
| `GB10_LOAD_SYNC_UPLOAD` | `--load-sync-upload` | flag | off | spmd | load: synchronous uploads |
| `GB10_LOAD_WORKERS` | `--load-workers` | int | 8 | spmd | load worker threads |
| `GB10_LOOP_TRACE` | `--loop-trace` | flag | off | spmd | scheduler loop trace |
| `GB10_MAX_SEQ_LEN` | `--max-seq-len` | int | per mode | spmd | max context / KV depth (the DSV4 TP paths honour it as the override) |
| `GB10_MEM_TRACE` | `--mem-trace` | flag | off | local | 1 Hz unified-memory timeline |
| `GB10_MEM_WATCHDOG_GB` | `--mem-watchdog-gb` | number | 5 | local | host-memory watchdog floor in GB (0 = off) |
| `GB10_MOE_COOP` | `--moe-coop` | on/off | off | spmd | cooperative expert stream (tunable override) |
| `GB10_MOE_FH` | `--moe-fh` | on/off | off | spmd | fused per-expert Hadamards (tunable override) |
| `GB10_MOE_FOLD` | `--moe-fold` | flag | off | spmd | E12 fold the shared expert into the grouped MoE launches (opt-in) |
| `GB10_MOE_FOLD_DEBUG` | `--moe-fold-debug` | flag | off | spmd | E12 fold debug prints |
| `GB10_MOE_GROUPED_MIN` | `--moe-grouped-min` | int | built-in | spmd | NVFP4 grouped MoE min rows |
| `GB10_MOE_GROUPED_WIDE` | `--moe-grouped-wide` | on/off | on | spmd | NVFP4 wide grouped MoE |
| `GB10_MOE_GU_FOLD` | `--moe-gu-fold` | int | 6 (tuned) | spmd | fold xq_had_suh_multi into the WP20 gate/up prologue up to width N, 0..16 (tunable moe.gu_fold override) |
| `GB10_MOE_NATIVE_PF` | `--moe-native-pf` | flag | off | spmd | NVFP4 native MoE prefill |
| `GB10_MOE_NO_FOLD` | `--moe-no-fold` | flag | off | spmd | E12 fold escape (the separate-launch shared MLP) |
| `GB10_MOE_SHOVL` | `--moe-shovl` | on/off | on | spmd | shared-expert overlap on the WP20 path (tunable override) |
| `GB10_MOE_VARIANT` | `--moe-variant` | one of plain / u4 / x2 / rast / lb5 / lb4 / pdl | plain | spmd | NVFP4 MoE kernel variant |
| `GB10_MTP_COVER` | `--mtp-cover` | int | 0 | spmd | EXL3 MTP coverage probe layer count |
| `GB10_MTP_COVER_OUT` | `--mtp-cover-out` | path | /tmp/mtp_cover.csv | spmd | EXL3 MTP coverage probe CSV path |
| `GB10_MTP_DISPATCH_TRACE` | `--mtp-dispatch-trace` | flag | off | spmd | MTP dispatch trace |
| `GB10_MTP_DUMP` | `--mtp-dump` | path | unset | spmd | EXL3 MTP dump dir (graphs off) |
| `GB10_MTP_GARBAGE_DRAFT` | `--mtp-garbage-draft` | flag | off | spmd | test drill: garbage drafts (losslessness negative control) |
| `GB10_MTP_LOSSLESS_AUDIT` | `--mtp-lossless-audit` | flag | off | spmd | MTP losslessness audit |
| `GB10_MXFP4` | `--mxfp4` | flag | off | spmd | MXFP4-native serving mode (fp4 decode/verify GEMMs on the OMMA path) |
| `GB10_MXFP4_ALLOW_EXPERTS` | `--mxfp4-allow-experts` | flag | off | spmd | MXFP4: allow expert tensors |
| `GB10_MXFP4_ECONOMY` | `--mxfp4-economy` | flag | off | spmd | MXFP4 economy (drop the bf16 copies) |
| `GB10_MXFP4_FUSED` | `--mxfp4-fused` | flag | off | spmd | MXFP4 fused kernels |
| `GB10_MXFP4_FUSED_PREFILL` | `--mxfp4-fused-prefill` | on/off | on | spmd | MXFP4 fused prefill |
| `GB10_MXFP4_MTP_NATIVE` | `--mxfp4-mtp-native` | flag | off | spmd | MXFP4: native MTP head (allowlist escape) |
| `GB10_MXFP4_PREFILL` | `--mxfp4-prefill` | flag | off | spmd | MXFP4 native prefill |
| `GB10_MXFP4_PREFILL_CHECK` | `--mxfp4-prefill-check` | flag | off | spmd | MXFP4 prefill A/B check |
| `GB10_MXFP4_XCHAIN_CAPTURE` | `--mxfp4-xchain-capture` | flag | off | spmd | MXFP4 cross-chain capture hooks (the probes arm it) |
| `GB10_NO_ATTN_E` | `--no-attn-e` | flag | off | spmd | NVFP4: no attention-E path |
| `GB10_NO_DECODE_GRAPHS` | `--no-decode-graphs` | flag | off | spmd | disable decode CUDA graphs (diagnostic) |
| `GB10_NO_DF2_GRAPH` | `--no-df2-graph` | flag | off | spmd | DFlash2: no round graph (escape) |
| `GB10_NO_DF2_INJECT_GRAPH` | `--no-df2-inject-graph` | flag | off | spmd | DFlash2: no inject graph (escape) |
| `GB10_NO_DF2_Q2STAGE` | `--no-df2-q2stage` | flag | off | spmd | DFlash2: no q2 staging (escape) |
| `GB10_NO_FUSED_PERHEAD_ROPE` | `--no-fused-perhead-rope` | flag | off | spmd | NVFP4: unfused per-head rope (escape) |
| `GB10_NO_GQPACK` | `--no-gqpack` | flag | off | spmd | NVFP4: no GQA packing (escape) |
| `GB10_NO_QKV_VIEW` | `--no-qkv-view` | flag | off | spmd | NVFP4: old qkv pipeline (escape) |
| `GB10_NO_VERIFY_GRAPH` | `--no-verify-graph` | flag | off | spmd | NVFP4: no verify graph (escape; some probes set it) |
| `GB10_P1_TIME_ALL` | `--p1-time-all` | flag | off | spmd | EXL3 bench: time every chunk size |
| `GB10_PACKED_CACHE` | `--packed-cache` | flag | off | spmd | DSV4 packed fp4 cache |
| `GB10_PAIR_SEQ` | `--pair-seq` | flag | off | spmd | DSV4 sequential pair attention |
| `GB10_PF4_TRACE` | `--pf4-trace` | flag | off | spmd | MXFP4 prefill trace |
| `GB10_PF8_CHECK` | `--pf8-check` | flag | off | spmd | fp8 prefill check |
| `GB10_PF8_DUMP` | `--pf8-dump` | flag | off | spmd | fp8 prefill dump |
| `GB10_PF8_TRACE` | `--pf8-trace` | flag | off | spmd | fp8 prefill trace |
| `GB10_PFX1_HEADKV_POISON` | `--pfx1-headkv-poison` | flag | off | spmd | PFX1 head-KV poison drill |
| `GB10_PFX1_OFF` | `--pfx1-off` | text | unset | spmd | PFX1 escape (1/all/<parts>) |
| `GB10_PFX1_ROWS8_MAX` | `--pfx1-rows8-max` | int | built-in | spmd | PFX1 rows8 max rows |
| `GB10_PF_DUMP` | `--pf-dump` | flag | off | spmd | NVFP4 prefill dump |
| `GB10_PF_GDN_SCAN` | `--pf-gdn-scan` | on/off | on | spmd | chunkwise GDN prefill scan re-scheduled tc2 (tunable override) |
| `GB10_PF_HC_FUSE` | `--pf-hc-fuse` | on/off | on | spmd | prefill hc inject+norm / mix fused (tunable override) |
| `GB10_PF_MIXER4` | `--pf-mixer4` | text | unset | spmd | NVFP4 prefill 4-bit mixer set (safe / all) |
| `GB10_PF_MOE_FOLD` | `--pf-moe-fold` | on/off | on | spmd | prefill MoE glue folded into the expert epilogues (tunable override) |
| `GB10_PF_MOE_KERNEL` | `--pf-moe-kernel` | on/off | on | spmd | pipelined prefill MoE expert kernel (tunable override) |
| `GB10_PF_MOE_MT_PE` | `--pf-moe-mt-pe` | on/off | on | spmd | WP19 per-expert MT (tunable override) |
| `GB10_PF_MOE_PF2` | `--pf-moe-pf2` | int | 3 | spmd | prefill MoE smem trellis ring mask 0..15 (tunable override) |
| `GB10_PF_QKV_FAST` | `--no-pf-qkv-fast` | flag | off | spmd | NVFP4: disable the fast prefill qkv path |
| `GB10_PF_RM_NOCACHE` | `--pf-rm-nocache` | flag | off | spmd | NVFP4 prefill: no row-major cache |
| `GB10_PLE_OFFLOAD` | `--ple-offload` | one of ssd / none / off / gpu | none | spmd | qwen4_exp: keep the ~31 GB PLE n-gram table on SSD (ssd) or device-resident (none) |
| `GB10_PLE_RAM` | `--ple-ram` | one of on / off / auto | auto | spmd | EXL3 PLE table residency: RAM (on), per-token pread (off), auto = RAM iff >= 16 GiB stays free |
| `GB10_PLE_SSD_CACHE_ROWS` | `--ple-ssd-cache-rows` | int | 262144 | spmd | PLE SSD row cache capacity |
| `GB10_PLE_SSD_THREADS` | `--ple-ssd-threads` | int | 32 | spmd | PLE SSD reader threads |
| `GB10_PLE_TRACE` | `--ple-trace` | text | unset | spmd | PLE trace |
| `GB10_PLE_XCHECK` | `--ple-xcheck` | flag | off | spmd | PLE eager cross-check |
| `GB10_PQ8_FLASH` | `--pq8-flash` | one of 0 / off / 1 / on / causal | causal | spmd | PQ8 prefill flash kernel (0 = flash256, 1 = bitwise twin, causal = default) |
| `GB10_PQ8_GATHER` | `--pq8-gather` | on/off | on | spmd | PQ8 q8 prefill gather (tunable override) |
| `GB10_PQ8_KVW` | `--pq8-kvw` | on/off | on | spmd | PQ8 row-parallel KV writer (tunable override) |
| `GB10_PQ8_OFF` | `--pq8-off` | on/off | off | spmd | PQ8 escape (every PQ8 item off) |
| `GB10_PQ8_XCHECK` | `--pq8-xcheck` | flag | off | spmd | PQ8 eager cross-check |
| `GB10_PREFILLX_EXPECT_SIG` | `--prefillx-expect-sig` | text | unset | spmd | prefillx probe: expected signature |
| `GB10_PREFILLX_QSA_KEYS` | `--prefillx-qsa-keys` | flag | off | spmd | prefillx probe: dump QSA keys |
| `GB10_PREFILLX_REL` | `--prefillx-rel` | int or bare | off | spmd | prefillx probe: rel-L2 report (2 = verbose) |
| `GB10_PREFILLX_TIER` | `--prefillx-tier` | flag | off | spmd | prefillx probe: tier report |
| `GB10_PREFILLX_ULP` | `--prefillx-ulp` | text | unset | spmd | prefillx probe: ULP list |
| `GB10_PREFILLX_WIDTHS` | `--prefillx-widths` | text | unset | spmd | prefillx probe: widths list |
| `GB10_PREFILL_BF16_BINV` | `--prefill-bf16-binv` | on/off | on | spmd | NVFP4 bf16 prefill batch-invariance path |
| `GB10_PREFILL_REPS` | `--prefill-reps` | int | 1 | spmd | EXL3 prefill bench reps |
| `GB10_PREFILL_TRACE` | `--prefill-trace` | flag | off | spmd | prefill trace (every family) |
| `GB10_PREFILL_WARMUP` | `--prefill-warmup` | flag | off | spmd | EXL3 prefill bench warmup |
| `GB10_PREFIX_GRID_REUSE` | `--prefix-grid-reuse` | on/off | on | spmd | prefix-cache grid reuse |
| `GB10_PREFIX_TAIL_CKPT` | `--prefix-tail-ckpt` | one of 0 / 1 / 2 | 1 | spmd | tail (message-boundary) prefix checkpoints (tunable prefix.tail_ckpt override; owner default 1) |
| `GB10_PSK` | `--psk` | on/off | off | spmd | PSK opt-in (tunable override) |
| `GB10_PSK_G` | `--psk-g` | text | 0 (auto) | spmd | PSK persistent split-K grid per shape |
| `GB10_PSK_OFF` | `--psk-off` | on/off | off | spmd | PSK escape |
| `GB10_PSK_PF` | `--psk-pf` | int | 1 | spmd | PSK trellis L2 prefetch distance 0..127 |
| `GB10_PSK_XCHECK` | `--psk-xcheck` | flag | off | spmd | PSK eager cross-check |
| `GB10_Q4_DENSE_ATTN` | `--q4-dense-attn` | flag | off | spmd | NVFP4 q4 dense attention |
| `GB10_QSA_DUMP` | `--qsa-dump` | flag | off | spmd | NVFP4 QSA dumps |
| `GB10_QSA_DUMP_LAYER` | `--qsa-dump-layer` | int | unset | spmd | EXL3 QSA dump layer |
| `GB10_QSA_EXL3` | `--qsa-exl3` | on/off | on | spmd | EXL3 QSA (off = dense attention) |
| `GB10_QSA_POOL` | `--qsa-pool` | on/off | on | spmd | EXL3 QSA pooling |
| `GB10_QSA_ROWDUMP_LAYER` | `--qsa-rowdump-layer` | int | unset | spmd | EXL3 QSA row dump layer |
| `GB10_QSA_SCALAR` | `--qsa-scalar` | flag | off | spmd | NVFP4 scalar QSA |
| `GB10_QSA_SELECT` | `--qsa-select` | on/off | on | spmd | latency-lean QSA top-k select asc3 (tunable override) |
| `GB10_QSA_SEL_V1` | `--qsa-sel-v1` | flag | off | spmd | NVFP4 QSA select v1 |
| `GB10_QSA_TIME` | `--qsa-time` | flag | off | spmd | QSA timing (syncs) |
| `GB10_RDMA_DEV` | `--rdma-dev` | text | rocep1s0f1 | local | RDMA device(s): <rail1>[,<rail2>] |
| `GB10_REPRIME_XCHECK` | `--reprime-xcheck` | flag | off | spmd | re-prime eager cross-check |
| `GB10_ROPE_YARN_FACTOR` | set by `--rope-yarn-factor` | internal | 1.0 | spmd | trunk YaRN rope factor (set by --rope-yarn-factor) |
| `GB10_ROUND_TRACE` | `--round-trace` | flag | off | spmd | DFlash2 round trace |
| `GB10_ROUTER_COAL` | `--router-coal` | on/off | on | spmd | router fold with coalesced weight staging (tunable override) |
| `GB10_SPEC_PASS` | `--spec-pass` | text | default | spmd | spec-pass mode (diagnostic) |
| `GB10_SPEC_PASS_Q` | `--spec-pass-q` | number | built-in | spmd | spec-pass q |
| `GB10_SPEC_PASS_SKEW` | `--spec-pass-skew` | int | unset | spmd | spec-pass skew |
| `GB10_SPEC_PASS_XCHECK` | `--spec-pass-xcheck` | flag | off | spmd | spec-pass eager cross-check |
| `GB10_SPEC_RATIO_DHEAD` | `--spec-ratio-dhead` | on/off | on | spmd | real-q (ratio) draft passes on DHEAD (tunable spec.ratio_dhead override; owner default on) |
| `GB10_STATE_TAP` | `--state-tap` | on/off | on | spmd | NVFP4 state tap |
| `GB10_STEP_TOPK_DIR` | `--step-topk-dir` | path | unset | spmd | EXL3 step top-k dump dir |
| `GB10_TAIL_FIXED` | `--tail-fixed` | flag | off | spmd | NVFP4 fixed tail |
| `GB10_TPC_DEBUG` | `--tpc-debug` | flag | off | spmd | TP config debug print |
| `GB10_TP_ACCEPT` | `--tp-accept` | int | unset | spmd | TP bench_accept depth (set = run the accept branch) |
| `GB10_TP_AGREE_DRILL` | `--tp-agree-drill` | int | unset | local | test drill: corrupt THIS rank's agree hash at step N (per box: set it on the node's own command line for a one-sided fault) |
| `GB10_TP_BATCH_PROBE` | `--tp-batch-probe` | int | unset | spmd | TP batch probe width |
| `GB10_TP_BINV_TP` | `--tp-binv-tp` | flag | off | spmd | TP batch-invariance probe (also --probe-binv-tp) |
| `GB10_TP_BOOT_DELAY` | `--tp-boot-delay` | text | unset | spmd | test drill: <rank>:<seconds> sleep before the boot rendezvous |
| `GB10_TP_CACHE` | `--tp-cache` | path | ~/.cache | local | node blob cache dir |
| `GB10_TP_CAPTURE` | `--tp-capture` | path | unset | spmd | TP debug capture path |
| `GB10_TP_DDS_PRIOR` | `--tp-dds-prior` | text | on | spmd | WP23 cost-guard prior at TP=2: on = 2.7 ms/draft, off = 6.0, or <ms> |
| `GB10_TP_DECODE_CTX` | `--tp-decode-ctx` | int | unset | spmd | TP DecodeCtx probe context |
| `GB10_TP_DEC_BLOCKS` | `--tp-dec-blocks` | int | 8 | spmd | EXL3 TP decode K2 grid blocks |
| `GB10_TP_DEC_GRECV` | `--tp-dec-grecv` | on/off | on | spmd | EXL3 TP decode GPU receive (off = cpu_done only) |
| `GB10_TP_DEC_XPORT` | `--tp-dec-xport` | on/off | on | spmd | EXL3 TP decode transport (off = the serial pair) |
| `GB10_TP_DFLASH` | `--tp-dflash` | flag | off | spmd | NVFP4 TP DFlash drafter one-shot generate |
| `GB10_TP_DH_SHARD` | `--tp-dh-shard` | on/off | on | spmd | EXL3 TP sharded MTP-head screen |
| `GB10_TP_DIAG` | `--tp-diag` | flag | off | spmd | TP transport diagnostics |
| `GB10_TP_EP_DEAL` | `--ep-deal` | one of interleave / contig / freq | interleave | spmd | EXL3 TP static expert deal |
| `GB10_TP_EP_DEAL_FILE` | `--tp-ep-deal-file` | path | committed table | spmd | --ep-deal freq: an alternative deal table |
| `GB10_TP_EP_HIST` | `--tp-ep-hist` | path | unset | spmd | EXL3 TP expert histogram dump |
| `GB10_TP_FP32_PARTIALS` | `--tp-fp32-partials` | flag | off | spmd | NVFP4 TP fp32 partials |
| `GB10_TP_GPU_RECV` | `--tp-gpu-recv` | on/off | auto (TpConfig) | spmd | NVFP4 TP GPU-direct all-reduce receive (on / off; unset = the shipped config) |
| `GB10_TP_GRAPH` | `--tp-graph` | flag | off | spmd | NVFP4 TP decode graphs |
| `GB10_TP_HEAD_PROOF` | `--tp-head-proof` | flag | off | spmd | NVFP4 TP head proof |
| `GB10_TP_HEAD_PROOF_FAULT` | `--tp-head-proof-fault` | flag | off | local | test drill: NVFP4 TP head-proof fault on THIS rank (per box) |
| `GB10_TP_KS_TARGET` | `--tp-ks-target` | int | auto | spmd | EXL3 TP split-K target |
| `GB10_TP_MTP` | set by `--mtp on` | internal | off | spmd | NVFP4 TP bench MTP (set by --mtp on) |
| `GB10_TP_MTP_DEPTH` | set by `--mtp-depth` | internal | unset | spmd | NVFP4 TP bench MTP depth (set by --mtp-depth) |
| `GB10_TP_ONESHOT` | `--tp-oneshot` | flag | off | spmd | TP=4 one-shot all-peers push |
| `GB10_TP_PREFILL_PAYLOAD` | `--tp-prefill-payload` | int | built-in | spmd | NVFP4 TP prefill all-reduce chunk cap (bytes) |
| `GB10_TP_PREFILL_XPORT` | `--tp-prefill-xport` | one of off / 0 / false / single / on / 1 / dual | dual | spmd | EXL3 TP prefill transport (off = serial pair, single = one rail, dual = both rails) |
| `GB10_TP_REDUCE_FUSE` | `--tp-reduce-fuse` | on/off | auto (TpConfig) | spmd | NVFP4 TP fused reduce+residual+norm epilogue (on / off; unset = the shipped config) |
| `GB10_TP_ROUTE_FOLD` | `--tp-route-fold` | on/off | on | spmd | EXL3 TP route-ep fold into the router tail |
| `GB10_TP_SHARD_MIXERS` | set by `--tp (default) / --no-shard-mixers` | internal | on under TP | spmd | NVFP4 TP mixer sharding |
| `GB10_TP_SHARD_MTP` | `--tp-shard-mtp` | on/off | on | spmd | NVFP4 TP MTP-block sharding |
| `GB10_TP_SHARED_REPL` | `--tp-shared-repl` | flag | off | spmd | EXL3 TP replicated shared expert (diagnostic A/B) |
| `GB10_TP_SPIN_US` | `--tp-spin-us` | int | built-in | spmd | TP transport spin microseconds |
| `GB10_TP_STATE_TP` | `--tp-state-tp` | flag | off | spmd | TP GDN state probe |
| `GB10_TP_STEP_PROBE` | `--tp-step-probe` | int | unset | spmd | TP step probe |
| `GB10_TP_TAIL_DRILL` | `--tp-tail-drill` | flag | off | spmd | test drill: TP transport tail drill |
| `GB10_TP_TRACE` | `--tp-trace` | flag | off | spmd | TP trace |
| `GB10_TP_VP_HEAD` | `--tp-vp-head` | on/off | on | spmd | EXL3 TP vocab-parallel lm_head for greedy rows |
| `GB10_TP_XPORT_BLOCKS` | `--tp-xport-blocks` | int | 16 | spmd | EXL3 TP prefill transport blocks |
| `GB10_TP_XPORT_LOOKAHEAD` | `--tp-xport-lookahead` | int | 4 | spmd | EXL3 TP prefill transport epochs in flight per rail |
| `GB10_TRACE_DEQUANT` | `--trace-dequant` | flag | off | spmd | NVFP4 dequant trace |
| `GB10_TRACE_PREFILL` | `--trace-prefill` | flag | off | spmd | NVFP4 prefill trace |
| `GB10_TRACE_WIDE` | `--trace-wide` | flag | off | spmd | NVFP4 wide trace |
| `GB10_V2_DET` | `--v2-det` | flag | off | spmd | NVFP4 v2 determinism check |
| `GB10_VERIFY_SEQ` | `--verify-seq` | flag | off | spmd | DSV4 sequential verify |
| `GB10_W4A4_CHECK` | `--w4a4-check` | flag | off | spmd | W4A4 check |
| `GB10_W4A4_LMHEAD_NARROW` | `--w4a4-lmhead-narrow` | on/off | model config | spmd | W4A4 narrow lm_head (on / off; unset = the pack config) |
| `GB10_W4A4_N8` | `--w4a4-n8` | on/off | on | spmd | W4A4 narrow 8-row MMA kernel (off = the wide 128-row kernel) |
| `GB10_W4A4_PREFILL` | `--w4a4-prefill` | text | unset | spmd | W4A4 prefill groups (expert,mlp,attn,gdn) |
| `GB10_W4A4_TRACE` | `--w4a4-trace` | flag | off | spmd | W4A4 trace |
| `GB10_W4A4_VERIFY` | `--w4a4-verify` | text | unset | spmd | W4A4 verify groups (attn,mlp,gdn) |
| `GB10_W4DENSE` | `--w4dense` | on/off | off | spmd | W4/DENSE package opt-in |
| `GB10_W4DENSE_FIX` | `--w4dense-fix` | on/off | on | spmd | W4/DENSE split-K fixup (tunable override) |
| `GB10_W4DENSE_G` | `--w4dense-g` | text | tuned | spmd | W4/DENSE grid per shape |
| `GB10_W4DENSE_GSAT` | `--w4dense-gsat` | int | 36 | spmd | W4/DENSE G-picker saturation (tunable override) |
| `GB10_W4DENSE_MULTI` | `--w4dense-multi` | on/off | on | spmd | W4/DENSE attention groups (tunable override) |
| `GB10_W4DENSE_NST` | `--w4dense-nst` | one of 4 / 6 / 8 | 6 | spmd | W4/DENSE ring depth (tunable override) |
| `GB10_W4DENSE_OFF` | `--w4dense-off` | on/off | off | spmd | W4/DENSE escape |
| `GB10_W4DENSE_SILU` | `--w4dense-silu` | on/off | on | spmd | W4/DENSE silu-down (tunable override) |
| `GB10_W4DENSE_SUH` | `--w4dense-suh` | on/off | on | spmd | W4/DENSE fused suh (tunable override) |
| `GB10_W4DENSE_XCHECK` | `--w4dense-xcheck` | flag | off | spmd | W4/DENSE eager cross-check |
| `GB10_W4HC_FUSE` | `--w4hc-fuse` | on/off | on | spmd | W4/HC regridded mixer (tunable override) |
| `GB10_W4HC_G` | `--w4hc-g` | int | 96 | spmd | W4/HC grid target (tunable override) |
| `GB10_W4HC_INJ` | `--w4hc-inj` | on/off | on | spmd | W4/HC deferred inject + norm (tunable override) |
| `GB10_W4HC_LA` | `--w4hc-la` | int | 0 | spmd | W4/HC phase-A L2 look-ahead 0..7 (tunable override) |
| `GB10_W4HC_M1` | `--w4hc-m1` | on/off | on | spmd | W4/HC at m = 1 (tunable override) |
| `GB10_W4HC_MIX` | `--w4hc-mix` | on/off | on | spmd | W4/HC mix fused behind the mixer (tunable override) |
| `GB10_W4HC_OFF` | `--w4hc-off` | on/off | off | spmd | W4/HC escape (tunable override) |
| `GB10_W4HC_PF` | `--w4hc-pf` | one of 0 / 1 / 2 | 1 | spmd | W4/HC phase-B L2 prefetch mode (tunable override) |
| `GB10_W4HC_XCHECK` | `--w4hc-xcheck` | flag | off | spmd | W4/HC eager cross-check |
| `GB10_W4MOE` | `--w4moe` | flag | off | spmd | W4/MOE package opt-in |
| `GB10_W4MOE_G` | `--w4moe-g` | int | 0 (auto) | spmd | W4/MOE persistent grid pin (tunable override) |
| `GB10_W4MOE_MINM` | `--w4moe-minm` | int | 1 | spmd | W4/MOE min rows |
| `GB10_W4MOE_NST` | `--w4moe-nst` | one of 4 / 8 | 0 (the plan rule) | spmd | W4/MOE ring depth pin (tunable override) |
| `GB10_W4MOE_OFF` | `--w4moe-off` | text | unset | spmd | W4/MOE escape (1/all/gu,dn,fold) |
| `GB10_W4MOE_ORDER` | `--w4moe-order` | one of il / IL / 1 / 0 / seq | 0 (seq) | spmd | W4/MOE item order (il = interleaved; tunable override) |
| `GB10_W4MOE_SHOVL` | `--w4moe-shovl` | on/off | off | spmd | W4/MOE shared-expert overlap (tunable override) |
| `GB10_W4QSA` | `--w4qsa` | on/off | off | spmd | W4/QSA union gather opt-in |
| `GB10_W4QSA_OFF` | `--w4qsa-off` | on/off | off | spmd | W4/QSA escape |
| `GB10_W4QSA_XCHECK` | `--w4qsa-xcheck` | flag | off | spmd | W4/QSA eager cross-check |
| `GB10_W4S_OFF` | `--w4s-off` | text | unset | spmd | W4/SMALL escape (1/all/router,gdn,dattn,ple,draft) |
| `GB10_W4S_XCHECK` | `--w4s-xcheck` | text | unset | spmd | W4/SMALL eager cross-check (1/all/<parts>) |
| `GB10_WP01_OFF` | `--wp01-off` | flag | off | spmd | WP01 tokenizer cache escape |
| `GB10_WP01_XCHECK` | `--wp01-xcheck` | flag | off | spmd | WP01 tokenizer cache shadow check |
| `GB10_WP05_OFF` | `--wp05-off` | text | unset | spmd | WP05 escape (parts) |
| `GB10_WP05_XCHECK` | `--wp05-xcheck` | flag | off | spmd | WP05 eager cross-check |
| `GB10_WP09_ARGMAX` | `--wp09-argmax` | on/off | on | spmd | WP09 one-launch argmax (tunable override) |
| `GB10_WP09_OFF` | `--wp09-off` | on/off | off | spmd | WP09 umbrella escape (tunable override) |
| `GB10_WP09_PLE` | `--wp09-ple` | on/off | on | spmd | WP09 k-major PLE (tunable override) |
| `GB10_WP09_XCHECK` | `--wp09-xcheck` | flag | off | spmd | WP09 eager cross-check |
| `GB10_WP10_GEMV` | `--wp10-gemv` | on/off | on | spmd | WP10 GEMV lane remap (tunable override) |
| `GB10_WP10_OFF` | `--wp10-off` | on/off | off | spmd | WP10 routing-fold escape (tunable override) |
| `GB10_WP10_XCHECK` | `--wp10-xcheck` | flag | off | spmd | WP10 eager cross-check |
| `GB10_WP11_CVT` | `--wp11-cvt` | on/off | on | spmd | WP11 PRMT int8 convert (tunable override) |
| `GB10_WP11_OFF` | `--wp11-off` | on/off | off | spmd | WP11 umbrella escape (tunable override) |
| `GB10_WP11_R1` | `--wp11-r1` | one of 0 / 1 / 2 | 1 | spmd | WP11 R1 L2 prefetch mode (tunable override) |
| `GB10_WP11_R2` | `--wp11-r2` | on/off | on | spmd | WP11 R2 regridded mix (tunable override) |
| `GB10_WP11_R3` | `--wp11-r3` | on/off | on | spmd | WP11 R3 k-major q_up (tunable override) |
| `GB10_WP11_R4` | `--wp11-r4` | on/off | on | spmd | WP11 R4 hn staging (tunable override) |
| `GB10_WP12_OFF` | `--wp12-off` | text | unset | spmd | WP12 escape (1/all/step,diet,conv,ab) |
| `GB10_WP12_XCHECK` | `--wp12-xcheck` | flag | off | spmd | WP12 eager cross-check |
| `GB10_WP16_OFF` | `--wp16-off` | flag | off | spmd | WP16 prefix-cache checkpoint escape |
| `GB10_WP16_PROBE_TAIL` | `--wp16-probe-tail` | int | unset | spmd | WP16 probe tail |
| `GB10_WP16_XCHECK` | `--wp16-xcheck` | flag | off | spmd | WP16 eager cross-check |
| `GB10_WP18_OFF` | `--wp18-off` | flag | off | spmd | WP18 escape |
| `GB10_WP18_ROWS` | `--wp18-rows` | int | 256 | spmd | WP18 rows |
| `GB10_WP18_XCHECK` | `--wp18-xcheck` | flag | off | spmd | WP18 eager cross-check |
| `GB10_WP20_DIET` | `--wp20-diet` | on/off | on | spmd | WP20 word diet (tunable override) |
| `GB10_WP20_EPI` | `--wp20-epi` | text | unset | spmd | WP20 epilogue rung (tunable override) |
| `GB10_WP20_OFF` | `--wp20-off` | on/off | off | spmd | WP20 escape (tunable override) |
| `GB10_WP21_KS_OFF` | `--wp21-ks-off` | text | unset | spmd | WP21: listed ks keep the 3-launch tail |
| `GB10_WP21_OFF` | `--wp21-off` | flag | off | spmd | WP21 escape |
| `GB10_WP21_XCHECK` | `--wp21-xcheck` | flag | off | spmd | WP21 eager cross-check |
| `GB10_WP22R1_OFF` | `--wp22r1-off` | flag | off | spmd | WP22 R1 escape |
| `GB10_WP24_DUMP` | `--wp24-dump` | path | unset | spmd | WP24 sampler dump path |
| `GB10_WP24_XCHECK` | `--wp24-xcheck` | flag | off | spmd | WP24 sampler cross-check |
| `GB10_WP27_FORKS` | `--wp27-forks` | text | all | spmd | WP27 capture-DAG forks (all/none/<list>; tunable override) |
| `GB10_WP27_GRAPHS` | `--wp27-graphs` | text | all | spmd | WP27 graph classes (all/none/<list>; tunable override) |
| `GB10_WP27_OFF` | `--wp27-off` | on/off | off | spmd | WP27 escape |
| `GB10_WP27_PRIO` | `--wp27-prio` | on/off | off | spmd | WP27 node priorities (tunable override) |
| `GB10_XCHAIN_CTX_DUMP` | `--xchain-ctx-dump` | path | unset | spmd | NVFP4 cross-chain context dump path |
| `GB10_XCHAIN_CTX_LAYERS` | `--xchain-ctx-layers` | text | unset | spmd | NVFP4 cross-chain context layers |
| `GB10_XTP_FLOOR_SEEDS` | `--xtp-floor-seeds` | int | 6 | spmd | xtp floor probe seeds |
| `RUST_INFER_CPU_SAMPLE` | `--cpu-sample` | flag | off | spmd | sample on the CPU instead of the GPU |
| `RUST_INFER_DEQUANT_AT_LOAD` | `--dequant-at-load` | flag | off | spmd | dequantize weights at load |
| `RUST_INFER_DRAFT_VOCAB` | `--draft-vocab` | int | full | spmd | FR-Spec draft vocabulary subset size (0 = full vocab) |
| `RUST_INFER_DRAFT_VOCAB_FILE` | `--draft-vocab-file` | path | unset | spmd | FR-Spec draft vocabulary row file |
| `RUST_INFER_DUMP_PROMPT` | `--dump-prompt` | flag | off | head | server: dump rendered prompts |
| `RUST_INFER_DUMP_TOKENS` | `--dump-tokens` | flag | off | head | server: dump tokens |
| `RUST_INFER_DUMP_TOOLS` | `--dump-tools` | flag | off | head | server: dump tool parsing |
| `RUST_INFER_FAKE_QUANT` | `--fake-quant` | text | unset | spmd | fake-quant spec (layer:format,...) |
| `RUST_INFER_PREFILL_SCALAR` | `--prefill-scalar` | flag | off | spmd | scalar prefill path |
| `RUST_INFER_ZERO_KV` | `--zero-kv` | flag | off | spmd | zero the KV cache at alloc |
| `PP_HEAD_IP` | `--pp-head-ip` | text | unset | local | PP prefill node: the head IP to dial |
| `DSPARK_SYNTH_DIR` | `--dspark-synth-dir` | path | built-in | spmd | DSpark synthetic-table dir (offline tools) |
| `DFLASH2_SYNTH_DIR` | `--dflash2-synth-dir` | path | built-in | spmd | DFlash2 synthetic-table dir (offline tools) |
| `MTP_DRAFT_LOG` | `--mtp-draft-log` | path | unset | spmd | NVFP4 scheduler: per-step MTP draft log file |
| `MTP_CURVE_FILE` | `--mtp-curve-file` | path | unset | spmd | NVFP4 scheduler: MTP acceptance curve file |

## Removed without a registry option (merged into an existing flag, or dead)

| env var | use instead |
|---|---|
| `DSV4_BISECT_LEN` | --bisect-len <N> |
| `DSV4_MAX_SEQ_LEN` | --max-seq-len <N> |
| `DSV4_PREFILL_TRACE` | --prefill-trace |
| `GB10_DF2_STEP_DUMP` | --df2-step-dump <DIR> |
| `GB10_NODE_CHILD` | nothing (the node supervisor passes --once to its session child) |
| `GB10_WP27_TEST_LIST` | nothing (a unit-test-only knob, removed) |
| `RUST_INFER_GPU_SAMPLE` | nothing (GPU sampling is the default; --cpu-sample is the escape) |
| `RUST_INFER_MTP` | --mtp on|off |
| `RUST_INFER_MTP_STOCHASTIC` | --mtp on|off |
| `GB10_DFLASH_DIR` | --draft-dir <DIR> (the deprecated --dflash-dir spelling for the v1 source) |

## Kept (not engine options)

- `GB10_TEST_MODEL_DIR`, `GB10_TEST_TOKENIZER`: fixture paths read only inside `#[test]` functions
  (the `cargo test` harness has no command line of its own); the served binary never reads them.
- Third-party variables are untouched: `CUDA_*`, `NCCL_*`, `RUST_LOG`, `RUST_BACKTRACE`, `HOME`, `PATH`, ...
- Script-local variables (e.g. `BIN`, `MODEL`, `GB10_BENCH`, `GB10_GATE`) are shell variables of the
  harnesses, never read by the binary; the binary ignores them.

Also new: `--tp-prefill-overlap`, `--tp-vp-sampled`, `--tp-seq-parallel`, `--ep-deal` (existing flags)
are registry options now (shown by `--print-config`); `--ep-deal` gains `freq`.
