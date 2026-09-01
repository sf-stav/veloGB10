# qwen4_exp (Qwen3.8-Flash-Next) — validation scripts

Reference oracle and quantization checks for the `Family::Qwen4Exp` port. Everything Python runs
inside the `llmc-qwen4:latest` docker image (transformers 5.16 with native `qwen4_exp`); pass
`--user $(id -u):$(id -g)` so the files it writes are yours.

| File | Role |
|---|---|
| `gen_tiny.py` | Builds a tiny random qwen4_exp model in the checkpoint layout (`model.language_model.*`, sharded PLE table, root `config.json`). |
| `ref_forward.py` | HF forward (fp32, eager) on the **dequantized** weights of an engine NVFP4 artifact, teacher-forced on the engine's tokens; compares per-step logits with `--probe-q4 --dump-logits`. |
| `validate_quant.py` | Round-trip error of a quantized artifact vs its bf16 source (GEMM tensors + PLE records). |
| `nvfp4.py` | Shared NVFP4 / PLE-record decoders (mirror of `src/quant.rs`). |
| `ref_bf16.py` | Quality oracle: the HF reference on the ORIGINAL bf16 weights vs one or more engine dumps (`--probe-q4 --dump-logits`) — the served quantization error of each artifact (RTN / GPTQ / MR-GPTQ). |
| `compare_qsa.py` | Matches the engine's QSA selection lists (`GB10_QSA_DUMP=1` stderr) against the reference's (`ref_forward.py` under `QSA_DUMP=1`), with the block scores of any differing block. |
| `ref/` | The reference implementations the port was written against (HF `modeling_qwen4_exp.py`, SGLang `qwen4_exp*.py`, `hyperconnection.py`, `qwen4_ple_nvme.py`). |

## End-to-end check (tiny model, ~1 minute)

```bash
cd ~/workspace/veloGB10
IMG=llmc-qwen4:latest; U="--user $(id -u):$(id -g)"
docker run --rm $U -v $PWD/scripts/qwen4exp:/w -v ~/models:/models --entrypoint python $IMG /w/gen_tiny.py /models/q4-tiny-bf16
./target/release/gb10_inference --quantize --model-dir ~/models/q4-tiny-bf16 --out ~/models/q4-tiny-nvfp4 --recipe all
TOKS="17,45,300,7,3,99,120,121,5,60,61,62,63,3,200,201,202,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
./target/release/gb10_inference --probe-q4 --model-dir ~/models/q4-tiny-nvfp4 --tokens $TOKS --max-new-tokens 8 --max-seq-len 512 --dump-logits /var/tmp/dump-tiny
docker run --rm $U -v $PWD/scripts/qwen4exp:/w -v ~/models:/models -v /var/tmp/dump-tiny:/dump --entrypoint python $IMG /w/ref_forward.py /models/q4-tiny-nvfp4 /dump
```

Expected: `argmax agreement 8/8`, cosine ≥ 0.9999, relative L2 < 1% at every step (the residue is
bf16 activation rounding; the weights are byte-identical on both sides). The prompt deliberately
contains EOS tokens (id 3) so the PLE n-gram "EOS resets the context" rule is exercised. Add
`--ple-offload ssd` to the probe to run the SSD reader: its logits must be bit-identical to the
resident run (`cmp` the two `logits.f32`).

## QSA indexer check (tiny model with a 64-token budget, ~1 minute)

`gen_tiny.py <out> 64` builds the same tiny model with `indexer_budget=64` (sparse selection from 68
visible tokens), so a 128-token prompt exercises the indexer on both the prefill and the decode paths:

```bash
docker run --rm $U -v $PWD/scripts/qwen4exp:/w -v ~/models:/models --entrypoint python $IMG /w/gen_tiny.py /models/q4-tiny64-bf16 64
./target/release/gb10_inference --quantize --model-dir ~/models/q4-tiny64-bf16 --out ~/models/q4-tiny64-nvfp4 --recipe all
TOKS=$(python3 -c "import random; random.seed(7); t=[random.randint(4,511) for _ in range(128)]; t[40]=3; t[41]=3; print(','.join(map(str,t)))")
GB10_QSA_DUMP=1 ./target/release/gb10_inference --probe-q4 --model-dir ~/models/q4-tiny64-nvfp4 --tokens $TOKS \
    --max-new-tokens 8 --max-seq-len 512 --dump-logits /var/tmp/dump-tiny64 2> /var/tmp/qsa-eng.log
docker run --rm $U -e QSA_DUMP=1 -v $PWD/scripts/qwen4exp:/w -v ~/models:/models -v /var/tmp/dump-tiny64:/dump --entrypoint python $IMG /w/ref_forward.py /models/q4-tiny64-nvfp4 /dump
python3 scripts/qwen4exp/compare_qsa.py /var/tmp/qsa-eng.log /var/tmp/dump-tiny64/ref_sel.json
```

Expected: `argmax agreement 8/8`, cos ≥ 0.9998, and `selection sets identical: 7/8` — the one
differing position (the prefill's last query) is a genuine near-tie at the 16th block (reference
scores 1.014 vs 1.029, engine 1.022 vs 1.014; the engine's scores deviate from the fp32 reference by
~0.9 % on average, i.e. bf16 activation rounding). Control: `GB10_Q4_DENSE_ATTN=1` on the same run
gives relL2 ≈ 3 % instead of 0.7 %.

## Real-model checks

```bash
# quantization round-trip (reads the 336 GB source, ~2 min)
docker run --rm $U -v $PWD/scripts/qwen4exp:/w -v ~/models:/models --entrypoint python $IMG \
    /w/validate_quant.py /models/Qwen3.8-Flash-Next /models/Qwen3.8-Flash-Next-NVFP4-velo
# first light: prefill + greedy decode, no server
./target/release/gb10_inference --probe-q4 --model-dir ~/models/Qwen3.8-Flash-Next-NVFP4-velo --ple-offload ssd \
    --chat --prompt "Quelle est la capitale de la France ?" --max-new-tokens 64 --max-seq-len 2048
```

**Memory discipline on a single GB10** (two hard hangs on 2026-08-28): run `scripts/memlog.sh` in
the background, never set `GB10_LOAD_FORCE`, never run another GPU job or a docker container while
the big model loads or serves. The engine's watchdog (`GB10_MEM_WATCHDOG_GB`, default 5) exits the
process before the kernel's OOM path can freeze the box.
