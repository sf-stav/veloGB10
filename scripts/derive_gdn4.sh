#!/usr/bin/env bash
# Derive a gdn4 (GDN-nvfp4) model FROM an existing mixed (GDN-fp8) NVFP4 model — WITHOUT the bf16.
# Only the fp8 GDN in/out-proj tensors are re-quantized (fp8 -> bf16 -> nvfp4); everything else (already
# nvfp4, or bf16) is copied byte-for-byte. Output is a normal, fully-isolated model dir like any other.
# Streams shards (bounded RAM), so it works on the 397B too.
#
#   ./derive_gdn4.sh <mixed-model-dir> <gdn4-out-dir>
#   ./derive_gdn4.sh /mnt/models/3.5-397b_nvfp4 /mnt/models/3.5-397b_nvfp4-gdn4
#
# NB: only this direction is sound. You CANNOT derive mixed from gdn4 (4-bit -> 8-bit can't recover
# precision) — so keep the MIXED model as the master and produce gdn4 on demand with this script.
set -euo pipefail
SDIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$SDIR/gb10_inference}"; [ -x "$BIN" ] || BIN="$SDIR/../target/release/gb10_inference"
FROM="${1:?usage: derive_gdn4.sh <mixed-dir> <gdn4-out-dir>}"
OUT="${2:?usage: derive_gdn4.sh <mixed-dir> <gdn4-out-dir>}"
[ -f "$FROM/model.safetensors" ] || [ -f "$FROM/model.safetensors.index.json" ] || {
  echo "ERROR: $FROM is not a quantized model dir"; exit 1; }
echo "Deriving gdn4 from $FROM -> $OUT"
exec "$BIN" --requant-gdn --from "$FROM" --out "$OUT"
