#!/bin/bash
# Test for scripts/run_tp_server.sh (the public TP head launcher).
#
# The launcher decides the world size and the peer list the engine boots with, so its argv is
# asserted literally through --dry-run (no GPU, no model load, no network). It runs a COPY of the
# launcher in a scratch directory next to a stand-in binary that only leaves a marker when it is
# executed, so the test also proves a dry run starts nothing. Takes no arguments. Run:
#   ./scripts/test_run_tp_server.sh
set -uo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)/run_tp_server.sh"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
cp "$SRC" "$T/run_tp_server.sh"
printf '#!/bin/bash\ntouch "%s/EXECUTED"\n' "$T" > "$T/gb10_inference"
chmod +x "$T/run_tp_server.sh" "$T/gb10_inference"
mkdir "$T/model" && echo '{}' > "$T/model/config.json"
L="$T/run_tp_server.sh"
fail=0

# ok <label> <expected argv tail> <launcher args...>: the dry run succeeds and prints the argv
ok() {
  local label="$1" want="$2"; shift 2
  local out rc
  # the dry run prints a shell-quoted line (printf %q writes a comma as \,): compare it unquoted
  out=$("$L" --model-dir "$T/model" --dry-run "$@" 2>&1 | sed 's/\\,/,/g'); rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ] && printf '%s' "$out" | grep -qF -- "$want"; then echo "  ok   $label"
  else echo "  FAIL $label (exit $rc, wanted: $want)"; echo "       got: $out"; fail=1; fi
}
# refuse <label> <needle in the error> <launcher args...>: exit 2 with a message that says why
refuse() {
  local label="$1" want="$2"; shift 2
  local out rc
  out=$("$L" --model-dir "$T/model" --dry-run "$@" 2>&1); rc=$?
  if [ $rc -eq 2 ] && printf '%s' "$out" | grep -qF -- "$want"; then echo "  ok   $label"
  else echo "  FAIL $label (exit $rc, wanted exit 2 and: $want)"; echo "       got: $out"; fail=1; fi
}

A=192.0.2.5:29500; B=192.0.2.6:29500; C=192.0.2.7:29500

# the world size is 1 + the number of PEERS, however they were typed
ok     "one --node -> TP=2"                 "--tp 2 --nodes $A"            --node $A
ok     "three --node flags -> TP=4"         "--tp 4 --nodes $A,$B,$C"      --node $A --node $B --node $C
ok     "one --node with commas -> TP=4"     "--tp 4 --nodes $A,$B,$C"      --node "$A,$B,$C"
ok     "commas and repeats mix"             "--tp 4 --nodes $A,$B,$C"      --node "$A,$B" --node $C
ok     "explicit --tp matches the commas"   "--tp 4 --nodes $A,$B,$C"      --tp 4 --node "$A,$B,$C"
ok     "explicit --tp 2"                    "--tp 2 --nodes $A"            --tp 2 --node $A
refuse "two peers in one flag: world 3"     "2 peer(s) given with --node make a world of 3"   --node "$A,$B"
refuse "--tp 2 with three comma peers"      "--tp 2 does not match the 3 peer(s)"            --tp 2 --node "$A,$B,$C"

# a malformed peer list is refused here, not handed to the engine
refuse "empty peer in the middle"           "has an empty peer"            --node "$A,,$B"
refuse "leading comma"                      "has an empty peer"            --node ",$A"
refuse "trailing comma"                     "has an empty peer"            --node "$A,"
refuse "empty --node value"                 "has an empty peer"            --node ""
refuse "space instead of a comma"           "contains a space"             --node "$A $B $C"
refuse "comma then space"                   "contains a space"             --node "$A, $B, $C"
refuse "a shell pattern is not a peer"      "shell pattern character"      --node '*'
refuse "a pattern inside a list"            "shell pattern character"      --node "$A,192.0.2.[67]:29500"

# --tp is a plain decimal
refuse "--tp 02"                            "without leading zeros (got '02')"   --tp 02 --node $A
refuse "--tp 004"                           "without leading zeros (got '004')"  --tp 004 --node "$A,$B,$C"
refuse "--tp two"                           "--tp needs a whole number"          --tp two --node $A
refuse "--tp 4 with one peer"               "--tp 4 does not match the 1 peer(s)" --tp 4 --node $A
refuse "--tp 0"                             "--tp 0 does not match"              --tp 0 --node $A
refuse "--tp '' (an empty value)"           "--tp needs a whole number (got '')" --tp "" --node $A
refuse "--tp far out of range"              "--tp 99999999999999999999 does not match the 1 peer(s)" --tp 99999999999999999999 --node $A
# ...and the refusal is the launcher's two lines only (bash's own "integer expression expected" used to precede them)
out=$("$L" --model-dir "$T/model" --dry-run --tp 99999999999999999999 --node $A 2>&1)
if [ "$(printf '%s\n' "$out" | wc -l)" -eq 2 ] && ! printf '%s' "$out" | grep -q "integer expression"; then echo "  ok   an out-of-range --tp prints the launcher's message only"
else echo "  FAIL an out-of-range --tp prints more than the launcher's message"; echo "       got: $out"; fail=1; fi

# the fixed part of the argv, and the pass-through
ok     "serves on 0.0.0.0:9000"             "--server --host 0.0.0.0 --port 9000 --model-dir" --node $A
ok     "--port / --max-seq-len / --max-batch" "--port 9100 --model-dir $T/model --tp 2 --nodes $A --max-seq-len 32768 --max-batch 4" \
       --node $A --port 9100 --max-seq-len 32768 --max-batch 4
ok     "flags after -- pass through"        "--nodes $A --prefix-cache off --mtp off" --node $A -- --prefix-cache off --mtp off

# --help prints the header block (and mentions the comma form), exit 0
out=$("$L" --help 2>&1); rc=$?
if [ $rc -eq 0 ] && printf '%s' "$out" | grep -qF -- "separated by commas"; then echo "  ok   --help documents the comma form"
else echo "  FAIL --help (exit $rc)"; echo "$out"; fail=1; fi

# no dry run and no refusal may have started the engine
if [ -e "$T/EXECUTED" ]; then echo "  FAIL a dry run or a refused command line executed the binary"; fail=1
else echo "  ok   nothing was executed"; fi
# ...and without --dry-run the same command line does reach it (the stand-in leaves its marker)
"$L" --model-dir "$T/model" --node "$A,$B,$C" >/dev/null 2>&1
if [ -e "$T/EXECUTED" ]; then echo "  ok   without --dry-run the launcher execs the binary"
else echo "  FAIL without --dry-run the binary was not executed"; fail=1; fi

if [ $fail -eq 0 ]; then echo "RUN_TP_SERVER TEST PASS"; else echo "RUN_TP_SERVER TEST FAIL"; fi
exit $fail
