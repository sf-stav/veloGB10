# Qwen 3.8 Flash-Next EXL3 — step-by-step setup

This guide walks through bringing up **Qwen3.8-Flash-Next** from an **EXL3 pack** on veloGB10, in two
deployment shapes: **single machine (TP=1)** and **TP=2**. It covers the required files on each
machine, the exact launch commands, and the log lines you should expect.

- **Target model:** `doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw` — https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw
- **Format:** EXL3 (ExLlamaV3 trellis), 3.05 bpw. About 85 GB, including a 32.6 GB n-gram table.
- **Full context:** launch with `--max-seq-len 262144` (the model's full 256K).
- **TP=1 and TP=2 only** on this path. Any other world size exits with `--tp 2 only`.

> **The EXL3 pack must be present on every machine, at the same path.** Unlike the NVFP4 models, the
> head does **not** ship the EXL3 weights to the peer: each machine loads its own local copy. A
> mismatch between the two copies is not detected — the two ranks simply disagree.

> **Text only.** There is no vision on the EXL3 path; image requests are rejected. (Vision remains
> available on the NVFP4 / Qwen3.8 27B path.)

---

## 1. The files you need

### Binary + PTX (every machine)

The engine is **the binary + a `src/ptx/` directory of kernel artifacts**. The binary loads the PTX
relative to its **current working directory**, so run it from a directory that contains both. Do not
mismatch a binary with foreign PTX — the engine refuses to start on a build-id mismatch.

On every machine (nodes included), the directory must look like this:

```
.
├── gb10_inference
└── src
    └── ptx
        ├── exl3_bench.ptx
        ├── fused_decode.ptx
        ├── gemm_nvfp4.ptx
        ├── gpu_batch.ptx
        ├── gpu_batch_b3.ptx
        ├── gpu_dflash.ptx
        ├── gpu_dspark.ptx
        ├── gpu_dsv4.ptx
        ├── gpu_dsv4_attn.ptx
        ├── gpu_dsv4_comp.ptx
        ├── gpu_kernels.ptx
        ├── gpu_mxfp4.ptx
        ├── gpu_mxfp4_moe.ptx
        ├── gpu_vision.ptx
        ├── gpu_w4a4.ptx
        ├── mxfp4_bench.ptx
        ├── rms_norm.ptx
        └── silu_gate.ptx
```

**Copy the binary and the whole `src/ptx/` directory to every machine.** The release tarball already
has this layout.

### The EXL3 pack (every machine)

Download the pack from `doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw` and put it at the **same path on
every machine** — for example `~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw`:

```
Qwen3.8-Flash-Next-EXL3-3.05bpw/
├── config.json
├── config.json.native
├── model.safetensors.index.json
├── model-00001-of-00007.safetensors
├── ...
├── model-00007-of-00007.safetensors
├── ngram_embedding.safetensors          (32.6 GB — the n-gram table)
├── mtp_hyper_connection_mixer_patch.safetensors
├── quantization_config.json
├── chat_template.jinja
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
└── generation_config.json
```

There is no drafter to download: **MTP speculative decoding is built into this model** and enabled
automatically.

---

## 2. Start the node (TP=2 only)

On the peer machine, ensure §1 is in place, then run:

```bash
./gb10_inference --node --port 29500
```

Expected output:

```
[node-resident] supervisor up on port 29500 — one process per head session; kill this process to stop the node
[node] <hostname> ready: discovery on UDP 29499, control on TCP 29500, cache ~/.cache/gb10_tp
```

At this point the node is waiting for the head to launch. The node needs **no other flags** and no
model path — but its own copy of the pack must be at the same path the head will use (§1).

For the TP=2 example below we assume the node is running at `192.168.177.12:29500`.

> **A single machine (TP=1) needs no node process.** Skip to §3 and use the TP=1 command.

---

## 3. Bring up the server on the head

Ensure the head directory has the binary + `src/ptx/`, and the EXL3 pack from §1. Run the binary from
the directory holding `gb10_inference` and `src/ptx/`.

### TP=2 (one node + head)

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw \
  --tp 2 \
  --nodes 192.168.177.12:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on
```

### Single machine (TP=1)

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on
```

### What you should see

While the pack loads you should see lines like:

```
[exl3-serve] <host>: loading EXL3 pack ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw (width ..., max_pos 262144, kv-cache ...)
[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)
[exl3-serve] MTP chain-verify ON (k=..., greedy + sampled lanes, ...)
```

When the server is ready to serve, it prints:

```
[exl3-serve] listening on 0.0.0.0:9000 (model id: Qwen3.8-Flash-Next-exl3-3.05bpw)
[exl3-serve] TP: scheduler pinned to core 9; spawned host workers use cpus 5-8,15-17
[exl3-serve] TP head scheduler thread cpu mask: 9
[exl3-serve] TP control: step-go over RDMA while live, TCP Step only with events (TP-E)
```

On the TP=2 path the head also confirms the peer:

```
[exl3-serve] TP node rank 1 READY (mirror armed)
```

Once the `listening on 0.0.0.0:9000` line appears, the server is up and you can connect with any
OpenAI-compatible client at `http://<head-ip>:9000/v1`.

**Loading takes a while on the first boot** — the pack is ~85 GB and includes the 32.6 GB n-gram
table. Subsequent loads are faster once the engine's caches exist.

---

## 4. Notes

- **This path is text only.** Image requests are rejected; vision is available on the NVFP4 /
  Qwen3.8 27B path only.
- **262,144 tokens is the maximum.** YaRN is not implemented, so there is no 1M context.
- **`--max-batch 1` is what was tested at TP=2.** A larger batch works and is hash-exact per lane,
  but the lanes take turns, so it does not raise aggregate throughput.
- **Prefix caching changes wording, not correctness.** A cached turn is not bit-identical to a cold
  one — reuse re-chunks the prefill. Pass `--prefix-tail-ckpt 0` if you need bit-identical resumes.
- **Seeded sampled requests are not byte-reproducible across runs.** Greedy decoding is
  bitwise-lossless with respect to non-speculative decoding: MTP never changes a greedy token.
- **If the TP=2 link fails mid-serve, the server stops** and must be restarted; there is no
  auto-restart.
- **A node that fails during boot fails the head loudly** — the head exits instead of hanging, so a
  refused peer is reported rather than waiting forever.
- **Cache management.** On the NVFP4 path the engine caches transferred model blobs at
  `~/.cache/gb10_tp`; see **MANAGING_CACHE.md**. The EXL3 path loads the pack from the path you pass
  to `--model-dir`, so keep that directory where you want it.
- **Environment variables are gone.** The engine reads none: leaving a `GB10_*` variable set refuses
  startup and names the replacement flag. The full env→flag table is in **docs/ENV_TO_FLAGS.md**.
