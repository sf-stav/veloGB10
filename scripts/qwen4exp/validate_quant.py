#!/usr/bin/env python
# Round-trip check of an engine NVFP4 artifact against its bf16 source: dequantize a few GEMM
# tensors and PLE record ranges, report the relative L2 error (NVFP4 RTN noise ≈ 9-10%).
#
#   docker run --rm --user $(id -u):$(id -g) -v $PWD/scripts/qwen4exp:/w -v ~/models:/models \
#       --entrypoint python llmc-qwen4:latest /w/validate_quant.py /models/Qwen3.8-Flash-Next /models/Qwen3.8-Flash-Next-NVFP4-velo
import sys, json, numpy as np
from nvfp4 import load_safetensors, dequant_packed, dequant_ple_rows, bf16_to_f32

src, out = sys.argv[1], sys.argv[2]
S, O = load_safetensors(src), load_safetensors(out)
names = [n[:-len('.weight_packed')] for n in O if n.endswith('.weight_packed')]
for stem in sorted(names)[:40]:
    key = stem + '.weight' if (stem + '.weight') in S else stem
    dt, shape, raw = S[key]
    m, k = O[stem + '.weight_packed'][1]; k *= 2
    rows = min(m, 2048)
    w = bf16_to_f32(np.asarray(raw[:rows * k * 2])).reshape(rows, k)
    deq = dequant_packed(O, stem, rows)
    print(f"{stem:80s} [{m},{k}] relL2={np.linalg.norm(deq - w) / np.linalg.norm(w):.4f}")
side = json.load(open(f'{out}/ple_ngram_nvfp4.json'))
print({k: v for k, v in side.items() if k != 'shard_global_scales'})
rec = np.memmap(f"{out}/{side['file']}", dtype=np.uint8, mode='r'); R = side['rows_per_shard']
for sidx in sorted({0, side['num_shards'] // 2, side['num_shards'] - 1}):
    name = f"{side['tensor_prefix']}.shard_{sidx}.weight"; dt, shape, raw = S[name]
    for r0 in [0, max(0, R - 1000)]:
        n = 1000
        w = bf16_to_f32(np.asarray(raw[r0 * 320:(r0 + n) * 320])).reshape(n, 160)
        deq = dequant_ple_rows(np.asarray(rec[(sidx * R + r0) * 96:(sidx * R + r0 + n) * 96]).reshape(n, 96), side['shard_global_scales'][sidx])
        print(f"PLE shard {sidx} rows {r0}..: relL2={np.linalg.norm(deq - w) / np.linalg.norm(w):.4f}")
