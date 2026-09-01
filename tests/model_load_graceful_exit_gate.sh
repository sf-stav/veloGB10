#!/usr/bin/env bash
# Model-load graceful-exit gate (owner directive 2026-08-27): a broken/stale/wrong-format model
# dir must exit with the graceful user-facing message (name path + cause + fix), NEVER with
# "memory allocation of ... failed" / "Aborted (core dumped)" / a panic. A VALID model must still
# load. This asserts the SUCCESS signal (the graceful message), not the absence of an error.
#
# Usage:
#   GB10_VALID_MODEL_DIR=/path/to/a/valid/model ./tests/model_load_graceful_exit_gate.sh
#
# Drives the release binary on .11. Each case runs detached with the log in /tmp; the poller greps
# the log for the success/forbidden marker, kills the instant it lands. Exit 0 = all green.
set -u
BIN="${GB10_BIN:-$(dirname "$0")/../target/release/gb10_inference}"
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
VALID="${GB10_VALID_MODEL_DIR:-}"
BASE="/tmp/model_load_fixtures"
LOGBASE=/tmp/model_load_gate
mkdir -p "$LOGBASE"

pkill -x gb10_inference 2>/dev/null; sleep 2   # clean GPU first

# ---- build the broken fixtures ----
python3 - "$BASE" <<'PY'
import json, os, struct, shutil, sys
BASE=sys.argv[1]
shutil.rmtree(BASE, ignore_errors=True); os.makedirs(BASE)
CONFIG={"model_type":"qwen3_5","text_config":{
  "hidden_size":512,"intermediate_size":2048,"num_hidden_layers":2,"num_attention_heads":4,
  "num_key_value_heads":2,"head_dim":128,"vocab_size":1000,"layer_types":["full_attention"],
  "max_position_embeddings":2048}}
def write_cfg(d):
    with open(os.path.join(d,"config.json"),"w") as f: json.dump(CONFIG,f)
def safetensors(path, tensors):
    header={}; off=0; data=b""
    for name,(dt,shape,b) in tensors.items():
        n=1
        for s in shape: n*=s
        assert len(b)==n*{"F32":4}[dt], (name,len(b),n)
        header[name]={"dtype":dt,"shape":list(shape),"data_offsets":[off,off+len(b)]}
        data+=b; off+=len(b)
    h=json.dumps(header,separators=(",",":")).encode(); h+=b" "*((-len(h))%8)
    with open(path,"wb") as f:
        f.write(struct.pack("<Q",len(h))); f.write(h); f.write(data)
# 1 empty (config only, no shards)
d=os.path.join(BASE,"empty"); os.makedirs(d); write_cfg(d)
# 2 truncated (8-byte header length prefix = ~2^61, the OOM repro)
d=os.path.join(BASE,"truncated"); os.makedirs(d); write_cfg(d)
with open(os.path.join(d,"model-00001.safetensors"),"wb") as f:
    f.write(struct.pack("<Q",2336927755350992246)); f.write(b"\x00"*64)
# 3 missing_shard (index references a shard that is not on disk)
d=os.path.join(BASE,"missing_shard"); os.makedirs(d); write_cfg(d)
with open(os.path.join(d,"model.safetensors.index.json"),"w") as f:
    json.dump({"weight_map":{"model.foo":"model-00099.safetensors"}},f)
# 4 wrong_format (valid shard, but a scheme the engine does NOT recognize -> F32 .weight, no packed)
d=os.path.join(BASE,"wrong_format"); os.makedirs(d); write_cfg(d)
safetensors(os.path.join(d,"model-00001.safetensors"),
            {"model.layers.0.mlp.gate_proj.weight":("F32",[8,16],bytes([0])*512)})
print("fixtures ok")
PY

run_case() {
  local name="$1" expect="$2"
  local log="$LOGBASE/$name.log"
  rm -f "$log"
  setsid nohup "$BIN" --server --model-dir "$BASE/$name" --port 8099 --max-seq-len 1024 \
      >"$log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 1 120); do
    if grep -qE "$expect" "$log" 2>/dev/null; then ok=1; break; fi
    if grep -qE "memory allocation of|Aborted \(core dumped\)|panicked at|SIGABRT|core dumped" "$log" 2>/dev/null; then ok=0; break; fi
    sleep 2
  done
  pkill -x gb10_inference 2>/dev/null
  # Assert the graceful message AND no forbidden crash marker.
  local has_msg=0; grep -qE "$expect" "$log" && has_msg=1
  local forbidden=$(grep -cE "memory allocation of|Aborted \(core dumped\)|panicked at|SIGABRT|core dumped" "$log")
  local verdict="FAIL"
  if [[ $ok -eq 1 && $has_msg -eq 1 && $forbidden -eq 0 ]]; then verdict="PASS"; fi
  printf "  [%s] %-14s %s   (message=%s forbidden=%s)\n" "$verdict" "$name" "$(grep -qE 'Error loading model from' "$log" && echo "graceful message present" || echo "NO message")" "$has_msg" "$forbidden"
  [[ "$verdict" == "PASS" ]]
}

echo "=== BROKEN DIRS -> graceful exit (never OOM/Aborted/panic) ==="
PASS=0; TOTAL=0
for spec in "empty:no .safetensors shards found" \
            "truncated:corrupt safetensors header length" \
            "missing_shard:index references shard" \
            "wrong_format:not the converted NVFP4 format"; do
  name="${spec%%:*}"; expect="${spec#*:}"
  TOTAL=$((TOTAL+1)); run_case "$name" "$expect" && PASS=$((PASS+1))
done

echo "=== VALID CONTROL (must still load) ==="
log="$LOGBASE/valid.log"; rm -f "$log"
setsid nohup "$BIN" --server --model-dir "$VALID" --port 8099 --max-seq-len 2048 --max-tokens 64 >"$log" 2>&1 &
V=$!
ok=0
for _ in $(seq 1 120); do
  if grep -qE "OpenAI-compatible server running" "$log" 2>/dev/null; then ok=1; break; fi
  if grep -qE "memory allocation of|Aborted \(core dumped\)|panicked at|Error loading model" "$log" 2>/dev/null; then ok=0; break; fi
  sleep 3
done
pkill -x gb10_inference 2>/dev/null
if [[ $ok -eq 1 ]]; then printf "  [PASS] valid          server booted (model loaded)\n"; PASS=$((PASS+1)); else printf "  [FAIL] valid          did not load:\n"; tail -6 "$log"; fi
TOTAL=$((TOTAL+1))

echo "=== SUMMARY: $PASS/$TOTAL green ==="
[[ $PASS -eq $TOTAL ]]
