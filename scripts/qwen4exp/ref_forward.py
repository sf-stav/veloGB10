#!/usr/bin/env python
# Reference oracle for the qwen4_exp port: the HF `Qwen4ExpForCausalLM` forward (fp32, eager
# attention) on the DEQUANTIZED weights of an engine NVFP4 artifact, teacher-forced on the engine's
# own tokens, compared step by step with the logits the engine dumped (`--probe-q4 --dump-logits`).
# Both sides therefore see IDENTICAL weights; the only differences are activation rounding.
#
#   docker run --rm --user $(id -u):$(id -g) -v $PWD/scripts/qwen4exp:/w -v ~/models:/models -v <dump>:/dump \
#       --entrypoint python llmc-qwen4:latest /w/ref_forward.py /models/q4-tiny-nvfp4 /dump
import sys, json, struct, os, numpy as np, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
from nvfp4 import dequant_packed, load_safetensors, bf16_to_f32, dequant_ple_table

art, dump = sys.argv[1], sys.argv[2]
T = load_safetensors(art)
tc = dict(json.load(open(f'{art}/config.json'))['text_config'])
for k in ['model_type', 'dtype']: tc.pop(k, None)
cfg = Qwen4ExpTextConfig(**tc)
cfg._attn_implementation = 'eager'   # the QSA indexer needs a materialized (float) mask
model = Qwen4ExpForCausalLM(cfg).to(torch.float32)
model.config._attn_implementation = 'eager'
sd = {}
for stem in {k[:-len('.weight_packed')] for k in T if k.endswith('.weight_packed')}:
    sd[stem + '.weight'] = torch.from_numpy(dequant_packed(T, stem))
for k, (dt, shape, raw) in T.items():
    if k.endswith(('.weight_packed', '.weight_scale', '.weight_global_scale')): continue
    if dt == 'BF16': sd[k] = torch.from_numpy(bf16_to_f32(np.asarray(raw))).reshape(shape)
    elif dt == 'I64': sd[k] = torch.from_numpy(np.asarray(raw).view(np.int64).copy()).reshape(shape)
    elif dt == 'F32': sd[k] = torch.from_numpy(np.asarray(raw).view(np.float32).copy()).reshape(shape)
side = json.load(open(f'{art}/ple_ngram_nvfp4.json'))
sd[side['tensor_prefix'] + '.weight'] = torch.from_numpy(dequant_ple_table(art, side))
msd = model.state_dict()
fixed = {}
for k, v in sd.items():
    nk = k.replace('model.language_model.', 'model.', 1)
    if '.mlp.experts.' in nk and nk.endswith('.weight'):
        nk = nk[:-len('.weight')]                      # stacked experts carry no .weight in HF
        v = v.reshape(msd[nk].shape)                   # the quantizer flattened [E, M, K] to [E*M, K]
    fixed[nk] = v
missing = [k for k in msd if k not in fixed]; unexpected = [k for k in fixed if k not in msd]
print("missing:", missing, "unexpected:", unexpected)
assert not missing, "reference model is missing tensors"
model.load_state_dict({k: v.to(msd[k].dtype) for k, v in fixed.items() if k in msd}, strict=False)
model.eval()
meta = json.load(open(f'{dump}/tokens.json'))
prompt, gen, V, steps = meta['prompt'], meta['generated'], meta['vocab'], meta['steps']
seq = prompt + gen[:-1]        # engine step k's logits sit at position len(prompt)-1+k
# QSA_DUMP=1: record every attention layer's selected token set per query position (the engine's
# GB10_QSA_DUMP=1 prints its lists; compare_qsa.py matches them).
if os.environ.get('QSA_DUMP'):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    sel_log = []
    _orig = M.Qwen4ExpTextQSAIndexer.forward
    def _hook(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        mask = _orig(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        sel = mask[0, 0] == 0 if mask.is_floating_point() else mask[0, 0]
        # Block scores, the reference formula (fp32 weights here), for the diagnosis of a differing block.
        import math
        cos, sin = position_embeddings
        L = hidden_states.shape[1]; R = self.compress_ratio
        qk = self.index_qk_proj(hidden_states)
        q, rk = torch.split(qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
        q = self.q_layernorm(q.reshape(1, L, -1, self.index_head_dim))
        q = M.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)[0]          # [L, H, D]
        rk = rk.reshape(1, L, self.index_head_dim)[0]
        nb = L // R
        pooled = self.k_layernorm(rk[:nb * R].reshape(nb, R, -1).float().mean(1))
        starts = torch.arange(nb) * R
        bk = M.apply_rotary_pos_emb(pooled.unsqueeze(0).unsqueeze(2), cos=cos[:, starts], sin=sin[:, starts], unsqueeze_dim=2)[0, :, 0]
        sc = torch.relu(torch.einsum('qhd,bd->qhb', q.float(), bk.float())).sum(1) / math.sqrt(self.index_head_dim)   # [L, nb]
        sel_log.append({'layer': self.layer_idx,
                        'sel': [torch.nonzero(sel[qi]).flatten().tolist() for qi in range(sel.shape[0])],
                        'scores': [sc[qi, :(qi + 1) // R].tolist() for qi in range(L)]})
        return mask
    M.Qwen4ExpTextQSAIndexer.forward = _hook
with torch.no_grad():
    ref = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0].float().numpy()
if os.environ.get('QSA_DUMP'):
    json.dump({'prompt_len': len(prompt), 'steps': steps, 'layers': sel_log}, open(f'{dump}/ref_sel.json', 'w'))
eng = np.fromfile(f'{dump}/logits.f32', dtype=np.float32).reshape(steps, V)
agree = 0
for k in range(steps):
    r, e = ref[len(prompt) - 1 + k], eng[k]
    cos = float(np.dot(r, e) / (np.linalg.norm(r) * np.linalg.norm(e) + 1e-9))
    rel = float(np.linalg.norm(r - e) / (np.linalg.norm(r) + 1e-9))
    ra, ea = int(r.argmax()), int(e.argmax()); agree += (ra == ea)
    print(f"step {k:2d}: cos={cos:.5f} relL2={rel:.4f} argmax ref={ra} eng={ea} {'OK' if ra == ea else 'DIFF'}"
          f"  ref top5={np.argsort(-r)[:5].tolist()} eng top5={np.argsort(-e)[:5].tolist()}")
print(f"argmax agreement {agree}/{steps}")
sys.exit(0 if agree == steps else 1)
