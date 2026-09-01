# Host-side decoders for the engine's NVFP4 artifacts (compressed-tensors layout + the PLE row
# records), shared by validate_quant.py and ref_forward.py. Mirrors src/quant.rs exactly:
#   w = e2m1(nibble) * e4m3(block scale) / global_scale     (reciprocal global-scale convention)
import json, struct, os, numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)

def e4m3(b):
    b = b.astype(np.uint16)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 7).astype(np.float32)
    return np.where(exp == 0, sign * (man / 8) * 2.0 ** -6,
                    sign * (1 + man / 8) * np.exp2(exp - 7).astype(np.float32)).astype(np.float32)

def e2m1(codes):
    v = E2M1[codes & 7]
    return np.where(codes & 8, -v, v)

def bf16_to_f32(raw):
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

def load_safetensors(model_dir):
    """{name: (dtype, shape, memmap bytes)} over every shard of the artifact."""
    idx_path = f'{model_dir}/model.safetensors.index.json'
    files = sorted(set(json.load(open(idx_path))['weight_map'].values())) if os.path.exists(idx_path) else ['model.safetensors']
    out = {}
    for f in files:
        path = f'{model_dir}/{f}'
        with open(path, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n)); base = 8 + n
        mm = np.memmap(path, dtype=np.uint8, mode='r')
        for k, t in h.items():
            if k == '__metadata__': continue
            a, b = t['data_offsets']
            out[k] = (t['dtype'], t['shape'], mm[base + a:base + b])
    return out

def dequant_packed(T, stem, rows=None):
    """Dequantize `{stem}.weight_packed` (+ scales, global scale) to a float32 [M, K] (or the first `rows`)."""
    dt, shape, qw = T[stem + '.weight_packed']; _, _, sc = T[stem + '.weight_scale']; _, _, gs = T[stem + '.weight_global_scale']
    gs = np.asarray(gs).view(np.float32)[0]; m, k2 = shape; k = k2 * 2
    m = m if rows is None else min(m, rows)
    q = np.asarray(qw[:m * k2]).reshape(m, k2)
    codes = np.stack([q & 0xF, q >> 4], -1).reshape(m, k)
    s = e4m3(np.asarray(sc[:m * (k // 16)]).reshape(m, k // 16)) / gs
    return (e2m1(codes) * np.repeat(s, 16, axis=1)).astype(np.float32)

def dequant_ple_rows(recs, gs):
    """96-B PLE records [n, 96] → float32 [n, 160] with the shard's reciprocal global scale."""
    n = recs.shape[0]
    q = recs[:, :80]
    codes = np.stack([q & 0xF, q >> 4], -1).reshape(n, 160)
    s = e4m3(recs[:, 80:90]) / gs
    return (e2m1(codes) * np.repeat(s, 16, axis=1)).astype(np.float32)

def dequant_ple_table(model_dir, side):
    rec = np.memmap(f"{model_dir}/{side['file']}", dtype=np.uint8, mode='r')
    R, N = side['rows_per_shard'], side['total_rows']
    tab = np.zeros((N, 160), dtype=np.float32)
    for i in range(side['num_shards']):
        tab[i * R:(i + 1) * R] = dequant_ple_rows(np.asarray(rec[i * R * 96:(i + 1) * R * 96]).reshape(R, 96), side['shard_global_scales'][i])
    return tab
