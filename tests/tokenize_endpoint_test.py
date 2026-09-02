#!/usr/bin/env python3
"""E2E gate for POST /v1/tokenize (vLLM-compatible de-facto shape) + serve no-regression.

Runs against a LIVE server (booted separately — the endpoint is pure tokenizer, the server still
needs the GPU to load the model):
    ./target/release/gb10_inference --server --model-dir /path/to/your/models/0.8b-nvfp4-mixed \
        --port 9377 --max-seq-len 8192
    python3 tests/tokenize_endpoint_test.py --port 9377 \
        --model-dir /path/to/your/models/0.8b-nvfp4-mixed [--model "Qwen/Qwen3.5-0.8B"]

Gates (PLAN/ADD_V1_TOKENIZE_PROMPT.md; every check asserts a SUCCESS signal):
  1. tokenize("Hello world") -> 200, count == len(tokens), ids == reference encode
     (TOKEN-IDENTITY: reference = HF tokenizers on the model's OWN tokenizer.json — the same
     source vLLM/transformers load).
  2. add_special_tokens true vs false — our ids match the reference for BOTH flags, and our
     differ/equal verdict matches the reference's (on Qwen3.5/Hy3/GLM the reference post-processor
     adds nothing, so true==false BY REFERENCE; the flag's wiring is proven by
     tests/tokenize_golden_test.rs on a specials-adding fixture).
  3. prompt as a token-id list echoes the same ids (round-trip).
  4. truncate_prompt_tokens caps the count to the LAST n (vLLM's left-truncation), no-ops when
     n >= len, 400 on n=0.
  5. errors: bad model / unsupported shape -> 400 with the engine's {"error":{...}} body;
     empty prompt -> 200 count 0 (vLLM behavior); GET /v1/tokenize -> 405.
  7. messages mode: count == usage.prompt_tokens of the SAME chat request (the client's
     template-overhead measurement, no hardcoded delta); multi-turn + truncate; mutual-exclusion
     and malformed-message 400s.
  8. detokenize: the client's example ids -> "Hello, world!", empty -> "", multi-byte
     tokenize->detokenize identity, skip_special_tokens default-false vs true, OOV/missing 400s,
     GET -> 405.
  9. over-length: >= max_seq_len -> 400 code context_length_exceeded; truncate_prompt_tokens
     rescues.
  6. no-regression: /v1/chat/completions still serves a non-empty completion on this build.
"""
import argparse, json, os, sys, urllib.error, urllib.request
import warnings
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9377)
    ap.add_argument("--model-dir", default=os.environ.get("GB10_TEST_MODEL_DIR", ""),
                    help="model dir; passed to the server for /v1/models id derivation")
    ap.add_argument("--model", default=None, help="served id; default: derived from the server")
    ap.add_argument("--max-model-len", type=int, default=None, help="expected max_model_len (the server's --max-seq-len)")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def post(path, body, timeout=300):
        req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try: return e.code, json.loads(raw)
            except Exception: return e.code, {"_raw": raw}

    def get(path, timeout=10):
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    # Served model id from /v1/models (what the contract says `model` must match).
    st, body = get("/v1/models")
    assert st == 200, f"/v1/models -> {st} (server not up?)"
    served = json.loads(body)["data"][0]["id"]
    if args.model: assert served == args.model, f"served id {served!r} != expected {args.model!r}"

    # Reference: the model's OWN tokenizer.json — the same artifact vLLM/transformers load.
    ref = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    def rtok(text, add_special): return ref.encode(text, add_special_tokens=add_special).ids

    results = []
    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    PROMPTS = ["Hello world", "The quick brown fox jumps over the lazy dog", "Hello, world! 你好 🚀"]

    # ---- Gate 1: tokenize -> 200, count == len(tokens), TOKEN-IDENTITY vs reference ----
    for text in PROMPTS:
        st, r = post("/v1/tokenize", {"model": served, "prompt": text})
        want = rtok(text, True)
        ok = (st == 200 and isinstance(r.get("tokens"), list)
              and r.get("count") == len(r["tokens"]) and r["tokens"] == want
              and isinstance(r.get("max_model_len"), int) and r["max_model_len"] > 0)
        check(f"gate1 tokenize {text!r} (identity+count)", ok,
              f"ids={r.get('tokens')} want={want} max_model_len={r.get('max_model_len')}")
    hw = post("/v1/tokenize", {"model": served, "prompt": "Hello world"})[1]

    # ---- Gate 2: add_special_tokens both ways vs reference ----
    for text in PROMPTS:
        st_t, r_t = post("/v1/tokenize", {"model": served, "prompt": text, "add_special_tokens": True})
        st_f, r_f = post("/v1/tokenize", {"model": served, "prompt": text, "add_special_tokens": False})
        want_t, want_f = rtok(text, True), rtok(text, False)
        ok = (st_t == 200 and st_f == 200 and r_t["tokens"] == want_t and r_f["tokens"] == want_f
              and (r_t["tokens"] != r_f["tokens"]) == (want_t != want_f))
        check(f"gate2 add_special_tokens true/false {text!r}", ok,
              f"ours differ={r_t['tokens'] != r_f['tokens']} reference differ={want_t != want_f}")

    # ---- Gate 3: token-id list echoes (round-trip) ----
    ids = hw["tokens"]
    st, r = post("/v1/tokenize", {"model": served, "prompt": ids})
    check("gate3 id-list round-trip", st == 200 and r.get("tokens") == ids and r.get("count") == len(ids),
          f"echo={r.get('tokens')} sent={ids}")
    # ids within vocab bounds only: one garbage id -> 400
    st_bad, r_bad = post("/v1/tokenize", {"model": served, "prompt": [ids[0], 10**9]})
    check("gate3b out-of-vocab id rejected", st_bad == 400, f"status={st_bad} body={r_bad}")

    # ---- Gate 4: truncate_prompt_tokens caps to the LAST n (vLLM left-truncation) ----
    long_text = PROMPTS[1]
    full = rtok(long_text, True)
    n = 3
    st, r = post("/v1/tokenize", {"model": served, "prompt": long_text, "truncate_prompt_tokens": n})
    check("gate4 truncate keeps LAST n", st == 200 and r["count"] == n and r["tokens"] == full[-n:],
          f"got={r['tokens']} want={full[-n:]}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": "Hello world", "truncate_prompt_tokens": 50})
    check("gate4b truncate >= len is a no-op", st == 200 and r["tokens"] == rtok("Hello world", True))
    st, r = post("/v1/tokenize", {"model": served, "prompt": "Hello world", "truncate_prompt_tokens": 0})
    check("gate4c truncate 0 -> 400", st == 400, f"status={st}")

    # ---- Gate 5: errors + 405 ----
    st, r = post("/v1/tokenize", {"model": "no-such-model", "prompt": "Hello world"})
    check("gate5 bad model -> 400 + error body",
          st == 400 and "error" in r and "not found" in r["error"]["message"], f"body={r}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": ""})
    check("gate5b empty prompt -> 200 count 0 (vLLM behavior)",
          st == 200 and r.get("count") == 0 and r.get("tokens") == [] and r.get("max_model_len", 0) > 0,
          f"status={st} body={r}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": []})
    check("gate5b2 empty id list -> 200 count 0", st == 200 and r.get("count") == 0, f"status={st}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": {"chunks": ["x"]}})
    check("gate5c unsupported prompt shape -> 400", st == 400 and "error" in r, f"status={st}")
    st, r = post("/v1/tokenize", {"model": served})
    check("gate5d missing prompt -> 400", st == 400 and "error" in r, f"status={st}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": "Hello world", "add_special_tokens": "yes"})
    check("gate5e non-bool add_special_tokens -> 400", st == 400, f"status={st}")
    code, _ = get("/v1/tokenize")
    check("gate5f GET /v1/tokenize -> 405", code == 405, f"status={code}")

    # ---- Gate 7: messages mode — count == usage.prompt_tokens of the SAME chat request ----
    # (the client's own measurement method: template-inclusive count with no hardcoded delta)
    chat_body = {"model": served, "messages": [{"role": "user", "content": "What is 2+2?"}],
                 "max_tokens": 1, "temperature": 0.0, "stream": False}
    st_c, r_c = post("/v1/chat/completions", chat_body, timeout=300)
    if st_c == 200 and r_c.get("usage"):
        usage_pt = r_c["usage"]["prompt_tokens"]
        st_m, r_m = post("/v1/tokenize", {"model": served,
                                          "messages": [{"role": "user", "content": "What is 2+2?"}]})
        check("gate7 messages count == usage.prompt_tokens",
              st_m == 200 and r_m.get("count") == usage_pt,
              f"messages={r_m.get('count')} usage.prompt_tokens={usage_pt}")
    else:
        check("gate7 messages count == usage.prompt_tokens", False, f"chat call failed: {st_c}")
    st_m, r_m = post("/v1/tokenize", {"model": served,
                                      "messages": [{"role": "user", "content": "Hi"},
                                                   {"role": "assistant", "content": None},
                                                   {"role": "user", "content": "st"}],
                                      "truncate_prompt_tokens": 10})
    check("gate7b messages multi-turn + truncate", st_m == 200 and r_m.get("count") == 10,
          f"status={st_m} count={r_m.get('count')}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": "Hi",
                                  "messages": [{"role": "user", "content": "Hi"}]})
    check("gate7c prompt AND messages -> 400", st == 400, f"status={st}")
    st, r = post("/v1/tokenize", {"model": served, "messages": [{"content": "no role"}]})
    check("gate7d malformed message -> 400", st == 400 and "error" in r, f"status={st}")

    # ---- Gate 8: detokenize — the decode half of the pair (exact-N prompt building) ----
    st, r = post("/v1/detokenize", {"model": served, "tokens": [9419, 11, 1814, 0]})
    check("gate8 detokenize client example -> 'Hello, world!'",
          st == 200 and r.get("prompt") == "Hello, world!" and r.get("model") == served,
          f"body={r}")
    st, r = post("/v1/detokenize", {"model": served, "tokens": []})
    check("gate8b empty tokens -> empty string", st == 200 and r.get("prompt") == "", f"body={r}")
    uni = "Hello, world! 你好 🚀"
    st_t, r_t = post("/v1/tokenize", {"model": served, "prompt": uni})
    st_d, r_d = post("/v1/detokenize", {"model": served, "tokens": r_t["tokens"]})
    check("gate8c tokenize->detokenize identity (multi-byte)",
          st_d == 200 and r_d.get("prompt") == uni, f"got={r_d.get('prompt')!r}")
    EOS = 248044  # <|endoftext|> (reference-derived; the engine's stop-token union lists it too)
    st, r = post("/v1/detokenize", {"model": served, "tokens": [EOS, 9419, 1814]})
    check("gate8d skip_special_tokens default false INCLUDES specials",
          st == 200 and r.get("prompt") == "<|endoftext|>Hello world", f"got={r.get('prompt')!r}")
    st, r = post("/v1/detokenize", {"model": served, "tokens": [EOS, 9419, 1814],
                                    "skip_special_tokens": True})
    check("gate8e skip_special_tokens=true drops specials",
          st == 200 and r.get("prompt") == "Hello world", f"got={r.get('prompt')!r}")
    st, r = post("/v1/detokenize", {"model": served, "tokens": [10**9]})
    check("gate8f OOV id -> 400", st == 400, f"status={st}")
    st, r = post("/v1/detokenize", {"model": served})
    check("gate8g missing tokens -> 400", st == 400, f"status={st}")
    code, _ = get("/v1/detokenize")
    check("gate8h GET /v1/detokenize -> 405", code == 405, f"status={code}")

    # ---- Gate 9: over-length -> distinguishable 400 (code context_length_exceeded) ----
    long_text = "hello world " * 6000   # ~12K tokens > the test server's 8192 max_seq_len
    st, r = post("/v1/tokenize", {"model": served, "prompt": long_text})
    check("gate9 over-length -> 400 context_length_exceeded",
          st == 400 and r.get("error", {}).get("code") == "context_length_exceeded",
          f"status={st} body={r}")
    st, r = post("/v1/tokenize", {"model": served, "prompt": long_text,
                                  "truncate_prompt_tokens": 100})
    check("gate9b truncate rescues over-length", st == 200 and r.get("count") == 100,
          f"status={st} count={r.get('count')}")

    # ---- Gate 6: existing-model serve round-trip (pure-add sanity on this build) ----
    st, r = post("/v1/chat/completions",
                 {"model": served, "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                  "max_tokens": 16, "temperature": 0.0, "stream": False}, timeout=300)
    content = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
    check("gate6 chat completion still serves", st == 200 and len(content.strip()) > 0,
          f"status={st} content={content.strip()[:60]!r}")

    # max_model_len contract: equals the server's effective --max-seq-len when provided
    if args.max_model_len is not None:
        check(f"max_model_len == {args.max_model_len}", hw.get("max_model_len") == args.max_model_len,
              f"got={hw.get('max_model_len')}")

    nfail = sum(1 for _, c in results if not c)
    print(f"\n=== TOKENIZE GATE SUMMARY === {len(results) - nfail}/{len(results)} PASS, {nfail} FAIL")
    sys.exit(1 if nfail else 0)

if __name__ == "__main__":
    main()
