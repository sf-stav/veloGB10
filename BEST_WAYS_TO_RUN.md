# Best ways to run — recommended launch recipes

Copy-paste launch lines for the models we have tuned on GB10. Each model gets a single-node
command plus TP=2 and TP=4 variants. These are the settings we actually run, not a maximal flag
list.

For a full step-by-step bring-up (node preparation, what the logs should say, cache management),
see [QWEN_27B_SETUP.md](QWEN_27B_SETUP.md) and [MANAGING_CACHE.md](MANAGING_CACHE.md).

**Models referenced here:**

| Role | Source |
|---|---|
| Qwen3.8 27B NVFP4 | [doth4580/Qwen3.8-27B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.8-27B-NVFP4-FULL) |
| Qwen3.8 27B FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| Qwen3.6 35B A3B NVFP4 | [doth4580/Qwen3.6-35B-A3B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.6-35B-A3B-NVFP4-FULL) |
| DFlash 2 draft for Qwen3.8 27B | [doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED](https://huggingface.co/doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED) |
| DFlash draft for Qwen3.6 35B A3B | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |

## Before you start

**TP=2 / TP=4 nodes.** On every machine that is not the head, start the node process first:

```bash
./gb10_inference --node --port 29500
```

**You do not need to copy models to the node machines.** The head ships the model, config, and
calibration over the network at sync time, and only the shards a node actually needs. See
[MANAGING_CACHE.md](MANAGING_CACHE.md) for the node-side cache.

**Two settings shape every recipe below:**

- **`--kv-cache k8v8` is single-lane.** It is rejected before the model loads unless
  `--max-batch 1`. That is why every recipe here runs `--max-batch 1`. To serve concurrent
  requests, switch to `--kv-cache bf16` (or `k8v4`) and raise `--max-batch` to the number of
  concurrent requests you want.
- **The draft model is passed with `--draft-dir`.** `--dflash-dir` still works as a deprecated
  alias, but it prints a warning on startup.

**Paths and addresses in these commands are examples.** Replace `~/models/...` with wherever you
keep your models, and the `192.168.177.x` addresses with your own node addresses.

---

## Qwen3.8 27B NVFP4

Model: [doth4580/Qwen3.8-27B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.8-27B-NVFP4-FULL).
Draft: [doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED](https://huggingface.co/doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED).

### Single node

```bash
./gb10_inference --server --model-dir ~/models/3.8-27b-nvfp4-full-all --kv-cache k8v8 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --prefix-cache on --default-presence-penalty 0.0 --mtp=auto --fp8-prefill on --reasoning-effort low --df2-block 16 --df2-round-shard on --spec-source dflash2 --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### TP=2

```bash
./gb10_inference --server --model-dir ~/models/3.8-27b-nvfp4-full-all --kv-cache k8v8 --tp 2 --nodes 192.168.177.12:29500 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --prefix-cache on --default-presence-penalty 0.0 --mtp=auto --fp8-prefill on --reasoning-effort low --df2-block 16 --df2-round-shard on --spec-source dflash2 --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### TP=4

```bash
./gb10_inference --server --model-dir ~/models/3.8-27b-nvfp4-full-all --kv-cache k8v8 --tp 4 --nodes 192.168.177.12:29500,192.168.177.13:29500,192.168.177.14:29500 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --prefix-cache on --default-presence-penalty 0.0 --mtp=auto --fp8-prefill on --reasoning-effort low --df2-block 16 --df2-round-shard on --spec-source dflash2 --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

---

## Qwen3.8 27B FP8

Model: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).
Same DFlash 2 draft as above.

### Single node

```bash
./gb10_inference --server --model-dir ~/models/Qwen/Qwen3.8-27B-FP8 --kv-cache k8v8 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --prefix-cache on --default-presence-penalty 0.0 --mtp=auto --fp8-prefill on --reasoning-effort low --df2-block 16 --df2-round-shard off --spec-source dflash2 --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### TP=2

```bash
./gb10_inference --server --model-dir ~/models/Qwen/Qwen3.8-27B-FP8 --kv-cache k8v8 --tp 2 --nodes 192.168.177.12:29500 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --prefix-cache on --default-presence-penalty 0.0 --mtp=auto --fp8-prefill on --reasoning-effort low --df2-block 16 --df2-round-shard off --spec-source dflash2 --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### TP=4

```bash
./gb10_inference --server --model-dir ~/models/Qwen/Qwen3.8-27B-FP8 --kv-cache k8v8 --tp 4 --nodes 192.168.177.12:29500,192.168.177.13:29500,192.168.177.14:29500 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --prefix-cache on --default-presence-penalty 0.0 --mtp=auto --fp8-prefill on --reasoning-effort low --df2-block 16 --df2-round-shard off --spec-source dflash2 --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

---

## Qwen3.6 35B A3B NVFP4

Model: [doth4580/Qwen3.6-35B-A3B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.6-35B-A3B-NVFP4-FULL).
Draft: [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash).
This is the DFlash v1 lane, selected with `--spec-source dflash`.

### Single node

```bash
./gb10_inference --server --kv-cache k8v8 --fp8-prefill off --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --default-presence-penalty 0 --prefix-cache on --mtp=auto --df2-block 16 --df2-round-shard on --model-dir ~/models/3.6-35b-nvfp4-mixed --spec-source dflash --draft-dir ~/models/Qwen3.6-35B-A3B-DFlash
```

### TP=2

```bash
./gb10_inference --server --kv-cache k8v8 --fp8-prefill off --tp 2 --nodes 192.168.177.12:29500 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --default-presence-penalty 0 --prefix-cache on --mtp=auto --df2-block 16 --df2-round-shard on --model-dir ~/models/3.6-35b-nvfp4-mixed --spec-source dflash --draft-dir ~/models/Qwen3.6-35B-A3B-DFlash
```

### TP=4

```bash
./gb10_inference --server --kv-cache k8v8 --fp8-prefill off --tp 4 --nodes 192.168.177.12:29500,192.168.177.13:29500,192.168.177.14:29500 --port 9000 --max-seq-len 262144 --max-batch 1 --max-tokens 65536 --default-presence-penalty 0 --prefix-cache on --mtp=auto --df2-block 16 --df2-round-shard on --model-dir ~/models/3.6-35b-nvfp4-mixed --spec-source dflash --draft-dir ~/models/Qwen3.6-35B-A3B-DFlash
```

---

## Confirming it came up

The server is ready when you see:

```
OpenAI-compatible server running on http://0.0.0.0:9000
Serving model: <model-name>  (GET /v1/models)
POST /v1/chat/completions   max_batch=1  default max_tokens=65536
```

On TP runs, one `[head] <ip> (rank N) READY` line per node confirms the nodes are wired up
correctly. The draft lane is live when you see `[df2] DFlash2 round RESIDENT`.

## Tuning the launch line

- **Concurrency:** `--max-batch <n>` accepts `n` concurrent requests. It requires dropping
  `--kv-cache k8v8` (see above), since int8 K/V rows are only read by the single-lane attention
  kernel. `--max-batch 1` with speculation is the fastest configuration for one user; above one
  lane, batching wins over speculation.
- **Context:** `--max-seq-len` is the KV budget. Lower it if you need the memory back; the
  recipes above use the full 256K the models advertise.
- **Reasoning:** `--reasoning-effort` accepts `no_think | low | medium | high | xhigh`. `low`
  shortens thinking on the Qwen3.8 templates; leave it unset to use the model's own default.
- **Presence penalty:** `--default-presence-penalty 0.0` turns the penalty off. The engine's
  built-in default is `1.5`, so the recipes above are explicitly overriding it; set a nonzero
  value to keep the repetition-dampening.
- **Draft block:** `--df2-block 16` drafts 16 tokens per round. `--df2-block 8` is the other
  supported width. The baked NVFP4 draft artifact above is block-agnostic: both widths read the
  same directory.
