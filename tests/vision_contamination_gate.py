import json, base64, sys, urllib.request, os, hashlib, re

# AGENTS §7 (owner rule): a CLI flag, never an env var — `vision_contamination_gate.py [--port N]`
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8099
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
RED="/tmp/vision_repro/solid_red.png"
BLUE="/tmp/vision_repro/solid_blue.png"

def b64(p):
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
def imgs(p):
    return [{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64(p)}"}},
            {"type":"text","text":"What color is this image? Answer with just the color name in one word."}]
def chat(messages, max_tokens=64, temp=0.0):
    # thinking off: the gate reads the answer, and a thinking preamble can exhaust max_tokens first
    body={"model":"x","messages":messages,"max_tokens":max_tokens,"temperature":temp,"stream":False,
          "chat_template_kwargs":{"enable_thinking":False}}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r:
        d=json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"]

def ans_text(content):
    # The model is verbose; extract the answer-ish portion. We just lower and keep it.
    return content.lower()

def has_color(t, color): return color in t

results=[]
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

# --- Scenario 1: FRESH conversations, two separate requests ---
a = chat([{"role":"user","content":imgs(RED)}])
la = ans_text(a)
b = chat([{"role":"user","content":imgs(BLUE)}])
lb = ans_text(b)
check("fresh A red->red", has_color(la,"red"), f"red in A={has_color(la,'red')} blue in A={has_color(la,'blue')}")
check("fresh B blue->blue", has_color(lb,"blue"), f"blue in B={has_color(lb,'blue')} red in B={has_color(lb,'red')}")
check("fresh B NOT red (contamination)", not has_color(lb,"red"), "red leaked into B = contamination")

# --- Scenario 2: SAME conversation, sequential images (the reported bug) ---
msgs=[{"role":"user","content":imgs(RED)}]
a1 = chat(msgs); la1=ans_text(a1)
a1_txt = a1.strip()
msgs.append({"role":"assistant","content":a1_txt})
msgs.append({"role":"user","content":imgs(BLUE)})
b1 = chat(msgs); lb1=ans_text(b1)
check("same turn1 red->red", has_color(la1,"red"), f"red={has_color(la1,'red')} blue={has_color(la1,'blue')}")
check("same turn2 blue->blue", has_color(lb1,"blue"), f"blue={has_color(lb1,'blue')} red={has_color(lb1,'red')}")
check("same turn2 NOT red (root bug)", not has_color(lb1,"red"), "red leaked into turn2 = contamination")

# --- Scenario 3: multi-image single turn (same request, red+blue) ---
m = chat([{"role":"user","content":imgs(RED)+imgs(BLUE)}]); lm=ans_text(m)
# It should mention both colors (or at least not just red-only / blue-only). Positive: both present.
hasr = has_color(lm,"red"); hasb = has_color(lm,"blue")
check("multi single-turn mentions both colors", hasr and hasb, f"red={hasr} blue={hasb}")

# --- Scenario 4: image-then-text ---
it = chat([{"role":"user","content":imgs(RED)}])
it2 = chat([{"role":"user","content":[{"type":"text","text":"What is 2+2?"}]}])
lit2 = ans_text(it2)
check("img->text text request OK", ("4" in lit2) or ("four" in lit2) or ("2+2" in lit2), f"got: {lit2[:60]}")

# --- Scenario 5: text-then-image ---
t2i = chat([{"role":"user","content":[{"type":"text","text":"What is 2+2?"}]}])
t2i2 = chat([{"role":"user","content":imgs(BLUE)}])
lt2i2 = ans_text(t2i2)
check("text->img blue->blue", has_color(lt2i2,"blue") and not has_color(lt2i2,"red"), f"blue={has_color(lt2i2,'blue')} red={has_color(lt2i2,'red')}")

# --- Scenario 6: determinism ---
def receipt(p):
    r1 = chat([{"role":"user","content":imgs(p)}])
    r2 = chat([{"role":"user","content":imgs(p)}])
    return (hashlib.sha256(r1.encode()).hexdigest(), hashlib.sha256(r2.encode()).hexdigest(), r1, r2)
rh1, rh2, r1, r2 = receipt(RED)
check("determinism red same sha", rh1==rh2, f"sha_a={rh1[:12]} sha_b={rh2[:12]}  equal={rh1==rh2}")

nfail = sum(1 for _,c,_ in results if not c)
print("\n=== GATE SUMMARY ===")
for n,c,d in results: print(f"  {'PASS' if c else 'FAIL'}  {n}")
print(f"TOTAL: {len(results)} checks, {nfail} failures")
sys.exit(1 if nfail else 0)
