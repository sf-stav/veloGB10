#!/bin/bash
# veloGB10 TP node (each peer box). Run this on the peer and forget it.
#
# The node needs NO options and NO model directory: the head ships everything at sync
# (model blobs to a content-addressed cache, the TP config, the MTP cost table, stop tokens).
# It is a RESIDENT supervisor: one clean child process per head session, and it re-arms by
# itself when the head goes away — you never restart it between runs. Kill the supervisor
# (pkill -x gb10_inference) to stop the node.
#
# Usage (v0.7.3: flags only — the old PORT=... environment overrides are GONE):
#   ./run_tp_node.sh [--port 29500] [--rdma-dev DEV] [--dry-run] [--help]
#
#   --port N       discovery/control port the head connects to   (default 29500)
#   --rdma-dev DEV RoCE device to use (default: auto — the head's discovery picks the rail)
#   --dry-run      print the exact engine command line and exit
#   --help         this text
set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SDIR/gb10_inference" ]; then cd "$SDIR"; BIN="./gb10_inference"          # deployed dir
else cd "$SDIR/.."; BIN="./target/release/gb10_inference"; fi                     # repo root
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN (build or stage it first)" >&2; exit 1; }

PORT=29500
RDMA_DEV=""
DRY_RUN=0

usage() { sed -n 's/^# \{0,1\}//p' "$0" | sed -n '3,13p'; }
die() { echo "ERROR: $*" >&2; echo "Try: $0 --help" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --port)     [ $# -ge 2 ] || die "--port needs a value"; PORT="$2"; shift 2 ;;
    --rdma-dev) [ $# -ge 2 ] || die "--rdma-dev needs a value"; RDMA_DEV="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  usage; exit 0 ;;
    *)          die "unknown option '$1' (the node takes no other options — the head ships everything)" ;;
  esac
done

ARGS=(--node --host 0.0.0.0 --port "$PORT")
[ -n "$RDMA_DEV" ] && ARGS+=(--rdma-dev "$RDMA_DEV")

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'dry-run engine command:\n  %s' "$BIN"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

echo "=== veloGB10 TP NODE — resident on port $PORT (zero config; head ships everything) ==="
exec "$BIN" "${ARGS[@]}"
