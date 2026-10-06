#!/bin/bash
# veloGB10 TP head (the serving box). OpenAI-compatible server driven SPMD across all ranks.
#
# Prereq: a node supervisor is running on every peer (./run_tp_node.sh). The head syncs the
# model + config to the peers (content-addressed cache — a re-run transfers nothing), brings
# up the RDMA link, runs the SPMD calibration, then serves. Output is bitwise in lockstep;
# the per-step agree() guard + watchdog abort both sides LOUDLY on divergence (that is a bug
# report, not a flake).
#
# Usage (v0.7.3: flags only — the old MODEL_DIR=/... environment overrides are GONE):
#   ./run_tp_server.sh --model-dir DIR --node PEER_IP:29500 [options] [-- <engine flags>]
#
# Options:
#   --model-dir DIR   (required) model directory on the head
#   --node HOST:PORT  (required) one per peer; repeat the flag for TP=4 (three peers)
#   --port N          serving port on the head            (default 9000)
#   --max-seq-len N   context length                     (default: pass-through to the engine)
#   --max-batch N     max concurrent requests            (default: pass-through to the engine)
#   --tp N            tensor-parallel world size         (default 2; must equal 1 + number of --node)
#   --dry-run         print the exact engine command line and exit
#   --help            this text
#   -- <flags>        everything after -- is passed to gb10_inference verbatim
#                     (e.g. -- --prefix-cache off --mtp=off --max-tokens 4096)
set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SDIR/gb10_inference" ]; then cd "$SDIR"; BIN="./gb10_inference"          # deployed dir
else cd "$SDIR/.."; BIN="./target/release/gb10_inference"; fi                     # repo root
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN (build or stage it first)" >&2; exit 1; }

MODEL_DIR=""
PORT=9000
SEQ=""
BATCH=""
TP=2
NODES=""
DRY_RUN=0

usage() { sed -n 's/^# \{0,1\}//p' "$0" | sed -n '3,26p'; }
die() { echo "ERROR: $*" >&2; echo "Try: $0 --help" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir)   [ $# -ge 2 ] || die "--model-dir needs a value"; MODEL_DIR="$2"; shift 2 ;;
    --node)        [ $# -ge 2 ] || die "--node needs a value (HOST:PORT)"; NODES="$NODES $2"; shift 2 ;;
    --port)        [ $# -ge 2 ] || die "--port needs a value"; PORT="$2"; shift 2 ;;
    --max-seq-len) [ $# -ge 2 ] || die "--max-seq-len needs a value"; SEQ="$2"; shift 2 ;;
    --max-batch)   [ $# -ge 2 ] || die "--max-batch needs a value"; BATCH="$2"; shift 2 ;;
    --tp)          [ $# -ge 2 ] || die "--tp needs a value"; TP="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --help|-h)     usage; exit 0 ;;
    --)            shift; break ;;
    *)             die "unknown option '$1'" ;;
  esac
done

[ -n "$MODEL_DIR" ] || die "--model-dir is required"
[ -n "$NODES" ] || die "--node is required (e.g. --node 192.0.2.5:29500)"
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: no model at $MODEL_DIR (no config.json)" >&2; exit 1; }

ARGS=(--server --host 0.0.0.0 --port "$PORT" --model-dir "$MODEL_DIR" --tp "$TP")
# repeated --node flags join into one comma-separated --nodes value (the engine's format)
NODES_CSV=""
for n in $NODES; do NODES_CSV="${NODES_CSV:+$NODES_CSV,}$n"; done
ARGS+=(--nodes "$NODES_CSV")
[ -n "$SEQ" ] && ARGS+=(--max-seq-len "$SEQ")
[ -n "$BATCH" ] && ARGS+=(--max-batch "$BATCH")
if [ $# -gt 0 ]; then ARGS+=("$@"); fi   # everything after -- passes through verbatim

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'dry-run engine command:\n  %s' "$BIN"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

echo "=== veloGB10 TP=$TP HEAD — $MODEL_DIR  port $PORT  nodes:$NODES ==="
echo "    (first start: model sync to the nodes + RDMA bring-up + SPMD calibration, a few minutes)"
exec "$BIN" "${ARGS[@]}"
