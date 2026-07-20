#!/bin/bash
# GB10 TP=2 — NODE (the second box). Run this on the PEER and forget it.
#
# The node needs NO env vars and NO model directory: the head ships everything at sync
# (model blobs to a content-addressed cache, the TP config, the MTP cost table, stop tokens).
# It is a RESIDENT supervisor: one clean child process per head session, and it re-arms by
# itself when the head goes away — you never restart it between runs. Kill the supervisor
# (pkill -x gb10_inference) to stop the node.
#
# Overrides:  PORT=29500  RDMA_DEV=rocep1s0f1  ./run_tp_node.sh
set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SDIR/gb10_inference" ]; then cd "$SDIR"; BIN="./gb10_inference"          # deployed dir
else cd "$SDIR/.."; BIN="./target/release/gb10_inference"; fi                     # repo root
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN (build or stage it first)"; exit 1; }

PORT=${PORT:-29500}
[ -n "${RDMA_DEV:-}" ] && export GB10_RDMA_DEV="$RDMA_DEV"

echo "=== GB10 TP=2 NODE — resident on port $PORT (zero config; head ships everything) ==="
exec "$BIN" --node --port "$PORT"
