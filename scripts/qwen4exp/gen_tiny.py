#!/usr/bin/env python
# Build a tiny random qwen4_exp (Qwen3.8-Flash-Next) text model in the CHECKPOINT layout the engine
# and the quantizer expect (`model.language_model.*`, sharded PLE n-gram table, root config.json with
# text_config). Runs inside the `llmc-qwen4` docker image (transformers 5.16 with native qwen4_exp):
#
#   docker run --rm --user $(id -u):$(id -g) -v $PWD/scripts/qwen4exp:/w -v ~/models:/models \
#       --entrypoint python llmc-qwen4:latest /w/gen_tiny.py /models/q4-tiny-bf16
#
# Then quantize with the engine and compare against ref_forward.py (see README.md).
import sys, json, os, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
from safetensors.torch import save_file

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
# Optional: a small indexer budget (e.g. 64 → sparse selection from 68 visible tokens) so short prompts
# exercise the QSA indexer against the HF reference; default 2048 (the real model's value).
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
torch.manual_seed(0)
# Geometry constraints of the engine's NVFP4 kernels: every GEMM M % 16 == 0 and K % 32 == 0;
# GDN value dim a multiple of 32; attention head_dim a multiple of 32. The PLE row dim is 160 (fixed
# by the quantizer's record codec) → ple_embed_dim = 16 heads * 160.
tc = dict(
    hidden_size=64, hc_count=4, hc_lowrank=32, num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    linear_num_key_heads=2, linear_num_value_heads=16, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
    shared_expert_intermediate_size=64, norm_topk_prob=True, vocab_size=512, rms_norm_eps=1e-6,
    max_position_embeddings=4096, eos_token_id=3, bos_token_id=3, pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                     "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    partial_rotary_factor=0.25, output_gate_type="sigmoid", hidden_act="silu", tie_word_embeddings=False,
    ple_layer_ids=[2], ple_embed_dim=2560, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=8,
    ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=128, seed=1234,
    indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=budget, indexer_compress_ratio=4,
    mamba_ssm_dtype="float32", attention_bias=False, attention_dropout=0.0, initializer_range=0.02,
    mtp_num_hidden_layers=0, split_ngram_parts=1,
)
cfg = Qwen4ExpTextConfig(**tc)
model = Qwen4ExpForCausalLM(cfg).to(torch.bfloat16)
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dtype != torch.bfloat16: continue
        f = torch.empty_like(p, dtype=torch.float32)
        if n.endswith('A_log'): p.copy_(torch.log(f.uniform_(0.5, 4)).to(torch.bfloat16))
        elif n.endswith('dt_bias'): p.copy_(f.uniform_(-1, 1).to(torch.bfloat16))
        elif 'norm' in n: p.copy_(f.uniform_(-0.3, 0.3).to(torch.bfloat16))
        elif n.endswith('conv1d.weight'): p.copy_(f.normal_().mul(0.3).to(torch.bfloat16))
        else: p.copy_(f.normal_().mul(0.05).to(torch.bfloat16))
sd = {}
for k, v in model.state_dict().items():
    if not (k.startswith('model.') or k.startswith('lm_head')): continue
    nk = k.replace('model.', 'model.language_model.', 1) if k.startswith('model.') else k
    if nk.endswith('ngram_embedding.weight'):
        sd[nk[:-len('.weight')] + '.shard_0.weight'] = v.contiguous()   # one shard = the whole table
        continue
    sd[nk] = v.contiguous()
print("tensors:", len(sd))
save_file(sd, f'{out}/model.safetensors', metadata={"format": "pt"})
json.dump({"weight_map": {k: "model.safetensors" for k in sd}, "metadata": {"total_size": 0}},
          open(f'{out}/model.safetensors.index.json', 'w'))
# The engine reads the HF-style layer_types ("full_attention"); the config object may have rewritten
# them to "qwen_sparse_attention" — write the original dict.
root = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "tie_word_embeddings": False,
        "text_config": {**tc, "model_type": "qwen4_exp_text", "dtype": "bfloat16"}}
json.dump(root, open(f'{out}/config.json', 'w'), indent=1)
print("wrote", out)
