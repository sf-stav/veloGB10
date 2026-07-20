#!/bin/bash
# GB10 TP=2 — HEAD (the serving box). OpenAI-compatible server driven SPMD across both ranks.
#
# Prereq: a node is running on the peer (./run_tp_node.sh). The head syncs the model + config
# to it (content-addressed cache — a re-run transfers nothing), brings up the RDMA link, runs
# the SPMD calibration, then serves. Output is bitwise in lockstep; the per-step agree() guard
# + watchdog abort both sides LOUDLY on divergence (that is a bug report, not a flake).
#
# Overrides:  MODEL_DIR=/path  PORT=9000  NODE=<peer-ip>:29500  SEQ=32768  BATCH=1
#             PREFIX=on|off  SHARD=on|off  MTP=on|off  ./run_tp_server.sh
# Examples:
#   MODEL_DIR=/models/3.6-27b-nvfp4-full NODE=10.0.0.2:29500 ./run_tp_server.sh
#   BATCH=4 SEQ=32768 MODEL_DIR=/models/3.5-122b-nvfp4-mixed NODE=10.0.0.2:29500 ./run_tp_server.sh
set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SDIR/gb10_inference" ]; then cd "$SDIR"; BIN="./gb10_inference"          # deployed dir
else cd "$SDIR/.."; BIN="./target/release/gb10_inference"; fi                     # repo root
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN (build or stage it first)"; exit 1; }

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR=/path/to/model}"
PORT=${PORT:-9000}
NODE="${NODE:?set NODE=<peer-ip>:29500 (the machine running run_tp_node.sh)}"
SEQ=${SEQ:-32768}
BATCH=${BATCH:-1}
PREFIX=${PREFIX:-on}
SHARD=${SHARD:-on}

[ ! -f "$MODEL_DIR/config.json" ] && { echo "ERROR: no model at $MODEL_DIR"; exit 1; }

ENV_ARGS=()
[ "$SHARD" = "on" ] && ENV_ARGS+=(GB10_TP_SHARD_MIXERS=1)
MTP_ARGS=()
[ -n "${MTP:-}" ] && MTP_ARGS=(--mtp="$MTP")

echo "=== GB10 TP=2 HEAD — $MODEL_DIR  port $PORT  node $NODE  seq $SEQ  batch $BATCH  prefix-cache $PREFIX  shard $SHARD ==="
echo "    (first start: model sync to the node + RDMA bring-up + SPMD calibration, a few minutes)"
exec env "${ENV_ARGS[@]}" "$BIN" --server \
  --model-dir "$MODEL_DIR" --tp --nodes "$NODE" --port "$PORT" \
  --max-seq-len "$SEQ" --max-batch "$BATCH" --max-tokens 4096 \
  --default-presence-penalty 1.5 --prefix-cache "$PREFIX" "${MTP_ARGS[@]}"
