#!/usr/bin/env python
# Quality oracle: the HF reference on the ORIGINAL bf16 weights (fp32 math), teacher-forced on the
# tokens of one or more engine dumps (`--probe-q4 --dump-logits`), reporting each dump's distance to
# the unquantized truth — i.e. the quantization error of that artifact as served by the engine.
#   docker run --rm -i --user $(id -u):$(id -g) -v $PWD/scripts/qwen4exp:/w -v ~/models:/models -v /var/tmp:/vt \
#       --entrypoint python llmc-qwen4:latest /w/ref_bf16.py /models/q4-tiny-bf16 /vt/dump-rtn /vt/dump-gptq
import sys, json, numpy as np, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
from safetensors.torch import load_file
import glob, os
src = sys.argv[1]; dumps = sys.argv[2:]
tc = dict(json.load(open(f'{src}/config.json'))['text_config'])
for k in ['model_type', 'dtype']: tc.pop(k, None)
cfg = Qwen4ExpTextConfig(**tc); cfg._attn_implementation = 'eager'
model = Qwen4ExpForCausalLM(cfg).to(torch.float32); model.config._attn_implementation = 'eager'
sd = {}
for f in sorted(glob.glob(f'{src}/*.safetensors')): sd.update(load_file(f))
msd = model.state_dict(); fixed = {}
ngram = {}
for k, v in sd.items():
    nk = k.replace('model.language_model.', 'model.', 1)
    if '.ngram_embedding.shard_' in nk:
        base = nk.split('.shard_')[0]; i = int(nk.split('.shard_')[1].split('.')[0]); ngram.setdefault(base, {})[i] = v; continue
    if '.mlp.experts.' in nk and nk.endswith('.weight'): nk = nk[:-len('.weight')]
    fixed[nk] = v
for base, parts in ngram.items(): fixed[base + '.weight'] = torch.cat([parts[i] for i in sorted(parts)], 0)
missing = [k for k in msd if k not in fixed]; print("missing:", missing[:5])
model.load_state_dict({k: v.to(msd[k].dtype) for k, v in fixed.items() if k in msd}, strict=False); model.eval()
for dump in dumps:
    meta = json.load(open(f'{dump}/tokens.json')); prompt, gen, V, steps = meta['prompt'], meta['generated'], meta['vocab'], meta['steps']
    seq = prompt + gen[:-1]
    with torch.no_grad(): ref = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0].float().numpy()
    eng = np.fromfile(f'{dump}/logits.f32', dtype=np.float32).reshape(steps, V)
    agree = 0; rels = []; kls = []
    for k in range(steps):
        r, e = ref[len(prompt) - 1 + k], eng[k]
        rels.append(float(np.linalg.norm(r - e) / (np.linalg.norm(r) + 1e-9)))
        pr = np.exp(r - r.max()); pr /= pr.sum(); pe = np.exp(e - e.max()); pe /= pe.sum()
        kls.append(float((pr * (np.log(pr + 1e-12) - np.log(pe + 1e-12))).sum()))
        agree += int(r.argmax() == e.argmax())
    print(f"{os.path.basename(dump):24s} argmax {agree}/{steps}  relL2 mean {np.mean(rels)*100:.2f}%  KL(ref||eng) mean {np.mean(kls):.4f}")
