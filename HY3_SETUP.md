# Tencent Hy3 (295B-A21B MoE) — step-by-step setup (TP=2)

This guide walks through bringing up the **Hy3** model on veloGB10 in a **two-node TP=2**
deployment. It covers the required files, the exact launch command, and the log lines you should
expect. Single-node and TP=4 covers are not documented yet; Hy3 support is still being stabilized
(see the roadmap).

- **Target model:** `doth4580/Tencent-Hy3-295B-A21B-NVFP4` — https://huggingface.co/doth4580/Tencent-Hy3-295B-A21B-NVFP4
- **Context:** launch with `--max-seq-len 262144` (the model's max).
- **Quantization:** `--mxfp4=on` (sm_121a OMMA path) with `--kv-cache q4`.

> **Note on model maturity.** Hy3 support is functional but still being stabilized. This guide is
> for the current working TP=2 path; treat it as the known-good configuration, not the final word.

---

## 1. The files you need

### Binary + PTX (every machine)

The engine is **the binary + a `src/ptx/` directory of kernel artifacts**. The binary loads the PTX
relative to its **current working directory**, so run it from a directory that contains both. The
binary is ~16 MB; the PTX files total ~12 MB. Do not mismatch a binary with foreign PTX.

On every machine (nodes included), the directory must look like this:

```
.
├── gb10_inference
└── src
    └── ptx
        ├── fused_decode.ptx
        ├── gemm_nvfp4.ptx
        ├── gpu_batch.ptx
        ├── gpu_batch_b3.ptx
        ├── gpu_dflash.ptx
        ├── gpu_dsv4.ptx
        ├── gpu_dsv4_attn.ptx
        ├── gpu_dsv4_comp.ptx
        ├── gpu_kernels.ptx
        ├── gpu_mxfp4.ptx
        ├── gpu_mxfp4_moe.ptx
        ├── mxfp4_bench.ptx
        ├── rms_norm.ptx
        └── silu_gate.ptx
```

**Copy the binary and the `src/ptx/` directory to each node machine.**

### The target model (head only)

You do **not** need to copy the target model to the node machines. The head ships what each node
needs automatically (see §3). Only the **head** needs the full Hy3 model directory. The Hy3 artifact
is large (~170 GB of shards), so on first sync the node receives ~170 GB of blobs.

---

## 2. Start the node

On the node machine, ensure the directory from §1 is in place (binary + `src/ptx/`), then run:

```bash
./gb10_inference --node --port 29500
```

Expected output:

```
[node-resident] supervisor up on port 29500 — one process per head session; kill this process to stop the node
[node] gx10-1dcd ready: discovery on UDP 29499, control on TCP 29500, cache ~/.cache/gb10_tp
```

At this point the node is waiting for the head to launch.

For the examples below we assume the node is running at:

- **TP=2:** `192.168.177.12:29500`

> **You do NOT need to copy the model to the node.** The head transfers the required model shards
> and files automatically. On first start this is a large sync (~170 GB for Hy3) and can take a
> while; subsequent starts are fast (only missing blobs are sent).

---

## 3. Bring up the server on the head

Ensure the head directory has the binary + `src/ptx/` and the Hy3 model directory. For this example
we assume the model lives at `~/models/hy3-nvfp4`.

The launch command is best written as a small script so the many options stay readable. The
following is the known-good TP=2 configuration for Hy3:

```bash
MODEL_DIR="${MODEL_DIR:-/path/to/your/models/hy3-nvfp4}"
PORT=${PORT:-9000}
NODE="${NODE:-<your node ip address>:29500}"
SEQ=${SEQ:-262144}
BATCH=${BATCH:-1}
PREFIX=${PREFIX:-off}
KVC=${KVC:-q4}
MTP=${MTP:-off}
# Fold explicitly OFF (belt + suspenders).
FOLD="--moe-no-fold"
# Graphs ON by default; GRAPHS=eager overrides to the non-graph path.
GRAPH_FLAGS=""
[ "${GRAPHS:-on}" = "eager" ] && GRAPH_FLAGS="--no-decode-graphs --no-verify-graph"

set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SDIR/gb10_inference" ]; then cd "$SDIR"; BIN="./gb10_inference"
else cd "$SDIR/.."; BIN="./target/release/gb10_inference"; fi
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: no model at $MODEL_DIR"; exit 1; }

echo "=== GB10 TP=2 HEAD — Hy3 NVFP4 MXFP4-ON  port $PORT  node $NODE  seq $SEQ  batch $BATCH  prefix-cache $PREFIX  mtp $MTP  kv-cache $KVC  fold off  graphs ${GRAPHS:-on} ==="
echo "    (first start: ~170 GB blob sync to the node; then ~85 s/rank load)"
exec "$BIN" --server \
  --model-dir "$MODEL_DIR" --tp --nodes "$NODE" --port "$PORT" \
  --max-seq-len "$SEQ" --max-batch "$BATCH" --max-tokens 65536 \
  --default-presence-penalty 1.5 --prefix-cache "$PREFIX" --mtp="$MTP" \
  --kv-cache "$KVC" --mxfp4=on ${FOLD} ${GRAPH_FLAGS}
```
> **Note (v0.7.0):** the engine no longer reads environment variables. Every option is a
> command-line flag; exporting one of the old `GB10_*` variables now refuses startup and names the
> replacement flag. The two lines above used to be `export GB10_MOE_NO_FOLD=1` and
> `export GB10_NO_DECODE_GRAPHS=1 GB10_NO_VERIFY_GRAPH=1`.

This is the configuration that was verified running at **~25 tok/s** on TP=2.

The key options for Hy3:

| Flag / env | Value | Why |
|---|---|---|
| `--tp` / `--nodes <ip>:29500` | on | Enable TP=2 serving against the node |
| `--kv-cache q4` | `q4` | 4-bit KV cache (Hy3 is a full-GQA model with large KV; q4 is needed to fit) |
| `--mxfp4=on` | on | Run the fp4 decode/verify GEMMs on the sm_121a OMMA path |
| `--mtp off` | off | Hy3 MTP currently works but is left off here |
| `--prefix-cache off` | off | Every request prefills its whole prompt (see the note below) |
| `--moe-no-fold` | on | MoE fold explicitly off |
| `GRAPHS=eager` (optional) | — | Override to the non-graph path |

### Memory: plan for a large footprint

Hy3 is a 295B-A21B model, so a single rank needs a large, sustained footprint. At `--max-seq-len
262144` with `--kv-cache q4`, the engine estimates:

- weights (per rank) ~80.2 GB
- KV cache (~2 slots) ~30.0 GB
- steady-state ~118.2 GB, startup peak ~125.7 GB

This is close to the ~121.6 GB physical limit of a single GB10. The engine prints a **WARNING** when
the estimate leaves little or no headroom, and notes that the `earlyoom` daemon SIGTERMs large
processes under memory pressure. If you see that warning:

- lower `--max-seq-len` (KV shrinks),
- keep `--kv-cache q4`,
- clear other large processes off the box before starting.

### What you should see

During the head bring-up you should see lines like:

```
[tp] config installed: world=2 shard_mixers=true shard_mtp=true graph=false ... mxfp4=true ...
[head] gx10-c9c4 — building manifest for ~/models/hy3-nvfp4 (world 2) ...
[head] manifest 'hy3-nvfp4': 19 artifacts, 169.59 GB
[head] 192.168.177.12 (rank 1) READY — model at ~/.cache/gb10_tp/models/hy3-nvfp4 (0.00 GB in 0.0s = 0.00 GB/s)
[head] shipped config to 192.168.177.12 (rank 1/2)
[head] 1 node(s) synced; all control streams RETAINED for the serving session
[tp] rank 0/2 — bringing up RDMA data-plane link on rocep1s0f1 (listening) ...
[tp] rank 0/2 — link UP
[tp] rank 0/2 — data-plane all-reduce link SANE (peer stamp 0xa1)
mxfp4-native mode ON: fp4 decode/verify GEMMs will run the sm_121a OMMA path (lossless load-time repack; bf16 chain preserved)
[load] total 201.2s | ptx+jit 0.0s rope 0.3s | shards: read 163.2s cpu 9.8s repack-inline 5.3s upload 4.3s | assemble 200.9s ...
Context: --max-seq-len 262144 (model max 262144). KV cache ~85.9 GB at batch 1.
Stop tokens: [120025, 120026, 120008]  (config.json advertises 120025)
HEAD (rank 0/2) — TP LINK UP (serving mode)
TP -- node rank 1 READY (mirror scheduler armed)
TP -- all 1 node(s) READY; binding HTTP
OpenAI-compatible server running on http://0.0.0.0:9000
Serving model: hy3-nvfp4  (GET /v1/models)
POST /v1/chat/completions   max_batch=1  default max_tokens=65536
```

Once you see the final three lines, the server is up and you can connect with any OpenAI-compatible
client.

**Checking the node connected properly** — look for the `READY` line listing your node IP (rank 1).
For TP=2 you should get one `READY` for rank 1.

---

## 4. Notes

- **First load is slower.** The engine creates caches on first load. For Hy3 the node also receives
  ~170 GB of blobs on first sync; the per-rank model load takes ~85 s after that.
- **Model transfer is automatic.** The head transfers model shards and config to the node on demand.
- **Cache management.** The engine caches transferred model blobs on the node at `~/.cache/gb10_tp`.
  To inspect or reclaim that cache, see **MANAGING_CACHE.md**.
- **Prefix cache is off by default here.** With `--prefix-cache off`, every request prefills its whole
  prompt (bit-exact, and slow on multi-turn agents: ~88% of prefill is recomputed). Enable it with
  `--prefix-cache on` if your workload benefits from prefix reuse.
- **Hy3 MLP head is bf16** (the bf16 lm_head is vocab-sharded under TP: 120832 → 60416 rows per rank).
