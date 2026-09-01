#!/bin/bash
# One chat completion against a running server, with wall time and tokens/s from `usage`.
#   scripts/qwen4exp/chat_test.sh [port] [prompt] [max_tokens]
PORT=${1:-9000}; PROMPT=${2:-"Quelle est la capitale de la France ? Réponds en une phrase."}; MAX=${3:-200}
BODY=$(python3 -c "import json,sys; print(json.dumps({'model':'x','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':int(sys.argv[2]),'temperature':0}))" "$PROMPT" "$MAX")
T0=$(date +%s.%N)
RESP=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")
T1=$(date +%s.%N)
python3 - "$RESP" "$T0" "$T1" <<'EOF'
import json, sys
raw, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try: d = json.loads(raw)
except Exception: print("NON-JSON RESPONSE:", raw[:500]); sys.exit(1)
u = d.get('usage', {}); m = d['choices'][0]['message']
dt = t1 - t0; ct = u.get('completion_tokens', 0)
print(f"wall {dt:.2f}s  prompt_tokens={u.get('prompt_tokens')} completion_tokens={ct}  ~{ct/dt:.1f} tok/s end-to-end")
if m.get('reasoning_content'): print("--- reasoning:", m['reasoning_content'][:600])
print("--- content:", (m.get('content') or '')[:800])
EOF
