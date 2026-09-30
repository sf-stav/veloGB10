#!/bin/bash
# GB10 TP=2 — NODE (the second box). Run this on the PEER and forget it.
#
# The node needs NO options and NO model directory: the head ships everything at sync
# (model blobs to a content-addressed cache, the TP config, the MTP cost table, stop tokens).
# It is a RESIDENT supervisor: one clean child process per head session, and it re-arms by
# itself when the head goes away — you never restart it between runs. Kill the supervisor
# (pkill -x gb10_inference) to stop the node.
#
# Overrides:  PORT=29500  RDMA_DEV=rocep1s0f1  TP_DIAG=1  ./run_tp_node.sh   (script variables; the
# binary gets flags — it reads no environment variables, AGENTS §7)
set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SDIR/gb10_inference" ]; then cd "$SDIR"; BIN="./gb10_inference"          # deployed dir
else cd "$SDIR/.."; BIN="./target/release/gb10_inference"; fi                     # repo root
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN (build or stage it first)"; exit 1; }

PORT=${PORT:-29500}
NODE_ARGS=()
[ -n "${RDMA_DEV:-}" ] && NODE_ARGS+=(--rdma-dev "$RDMA_DEV")
# R9 DIAGNOSTIC: --tp-diag rides the HEAD's TpConfig (the node installs the head's options), so a
# diag head drives a diag node; TP_DIAG=1 here is not needed and no longer exists as an env knob.

echo "=== GB10 TP=2 NODE — resident on port $PORT (zero config; head ships everything) ==="
exec "$BIN" --node --port "$PORT" "${NODE_ARGS[@]}"
