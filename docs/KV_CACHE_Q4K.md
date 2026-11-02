# q4k — a 4-bit packed KV cache on the EXL3 path

`--kv-cache q4k` stores both K and V at 4 bits on the q8 basis: the same H32 rotation, one f16
scale per 32-dim group, and two dims packed into one byte. A row is 144 B at head_dim 256
instead of q8's 272.

| | |
|---|---|
| Option | `--kv-cache q4k` (EXL3 packs only) |
| Format constant | `KV_Q4K = 5` (`XQ_KV_Q4K` on the device) |
| Reference | `main` @ `438176b` (v0.7.8), unreleased |
| Status | implemented, measured on one GB10; TP=2/4 not run |

## 1. Quality

Flash-Next 3.05 pack, 8 windows x 2,048 tokens (`--perplexity --window 2048 --max-windows 8`),
the same three texts the q8 reference uses:

| text | q8 | q4k | delta |
|---|---|---|---|
| code | 3.9768 | 3.9792 | +0.06 % |
| prose | 6.8758 | 6.8727 | -0.05 % |
| json | 1.4098 | 1.4098 | 0.00 % |

The probe's synthetic-row SNR is 22.98 dB (q8 47.07, the 4-bit `q4x` control 22.93). That is the
6 dB per bit the grid predicts, and the packed row lands where q4x does rather than below it.

## 2. Capacity

At 262,144 tokens on one 128 GB machine, four lanes:

| format | KV per lane | lanes at 262K | trunk KV at 262K |
|---|---|---|---|
| q8 | 5.61 GB | 5 | 13.69 GB |
| q4k | 3.76 GB | **8** | 7.25 GB |

The per-lane figure includes the sparse-attention indexer planes, which q4k does not touch.

## 3. Throughput

Short prompts (about 80 tokens, `--max-seq-len 8192`, `--prefix-cache off`, 128 output tokens,
best of 3). The QSA gathers and the W4S twins only run past the 2,051-token switch, so this
table prices the dense path alone:

| streams | q8 | q4k |
|---|---|---|
| 1 | 66.0 | 63.6 |
| 2 | 88.3 | 84.7 |
| 4 | 92.3 | 92.7 |

4-stream is weight-bandwidth bound and format-independent. The 1- and 2-stream gap is the 4-bit
unpack in the dense attention path.

5,312-token prompts, where `hg1` (lone decode), `union`/`hg3` (packed verify) and the pq8 prefill
gather all run:

| streams | q8 decode | q4k decode | q8 prompt | q4k prompt |
|---|---|---|---|---|
| 1 | 75.9 | **79.3** | 1622 | 1624 |
| 2 | 36.8 | **39.2** | 1207 | 1210 |

Past the switch, decode is 4.5 % to 6.5 % faster and prefill is unchanged.

## 4. Format

### 4.1 Row layout (hd = head_dim)

```
offset 0             : hd/2 bytes   packed 4-bit codes, 2 dims per byte
offset hd/2          : hd/32 * 2 B  f16 group scales, one per 32 dims
offset hd/2 + hd/16  : padding to a 16 B boundary
```

Dim `d` sits in byte `d/2` at bit `(d % 2) * 4`: even dims in the low nibble.

### 4.2 Sizes

| hd | q8 row | q4k row | ratio |
|---|---|---|---|
| 128 | 144 B | 80 B | 1.80x |
| 256 | 272 B | 144 B | 1.89x |

Bytes per element at hd 256: q8 8.5, q4k 4.5. The scale chunk starts at `hd/2`, 16 B aligned for
the head dims this engine serves (128 and 256).

### 4.3 Quantization

The rotation and the scale are q8's; only the grid and the packing are new:

```
v       = H32(x) * (1/sqrt(32))
s       = max over the 32-group of |v| + 1e-10      // stored as f16
code    = clamp(floor(v/s * 8 + 8), 0, 15)
dequant = (code - 7.5) * (s / 8)
```

Host and device run the same op order with the same `_rn` intrinsics, which is what
`--probe-exl3-kvq` checks bit for bit (`src/exl3_kvnll.rs::q4k_quant_row`).

### 4.4 Relation to q4x

`q4x` is the probe-only positive control: the same rotation, the same per-32 scale and the same
4-bit midpoint grid, but its codes live in q8's byte-per-dim row (`q4 * 16 + 8`), so the row
stays 272 B. q4k is that quantizer in half the row.

## 5. Where q4k runs

| path | how |
|---|---|
| prefill, dense rows (`flash256`) | own `a3fg_stage_v<XQ_KV_Q4K, ...>` case |
| decode and verify (`dense-v3`) | `xq_dv3_stage_v` takes the row's scale offset; `xq_dv3_ldv_q4k` unpacks the staged nibbles |
| QSA decode gather (`hg1`/`hg3`) and `union` | rows are staged whole, 16 B cp.async chunks driven by `rb`, and read through `xq_kv_ld8`: no kernel change |
| W4S draft attention | same: `xq_w4d_stage` is `rb`-driven and reads through `xq_kv_ld8` |
| W4S dense accumulate twin | `xq_w4a_*` takes the same `P4` constants and reader as dense-v3 |
| pq8 prefill gather | raw twin `xq_q4k_raw`/`xq_q4k_deq8` (4 code bytes + the group's f16 scale at `hd/2`) |
| lane packing | admitted (rows are symmetric); requires dense-v3, see §7 |

Escapes are unchanged: `--pq8-gather=0` = splitk4, `--w4s-off=dattn|draft` = the v3 and old
kernels, `--exl3-dense-v3=0` = the old group kernel.

## 6. Verification

Bitwise:

- `--probe-exl3-kvq --kv-cache q4k --model-dir <pack>`: rowbytes 144, device == host reference on
  bytes / deq / rec (0 mismatches), `ld8 == ld`, zero rows exact, SNR body 22.98 dB.
- `--exl3-dense-xcheck=1`: 0 mismatching halves on decode steps and lone verifies.
- `--probe-exl3-pack2` / `--probe-exl3-packstate`: PASS on q8 and q4k. q4k on the 3.05 pack:
  r_mean 1.48-1.51 over three runs, 0 live-state words differing on all three pairs (a single
  rep's speed ratio ranged 1.09-1.44, so quote the mean, not one rep).
- `--pq8-xcheck=1`: no MISMATCH over the three PPL texts.
- `--w4s-xcheck=dattn,draft` on q4k, graph capture off, `--log-level debug`: `dattn m=8 p0=405
  xq_attn_dense_acc4_w4 vs xq_attn_dense_acc4: bad 0 of 49152` and `draft m=1 dense_prep/splitk4_w4/
  combine_w4 vs prep/splitk4/combine: bad 0 of 408576`, no line with bad > 0. A short prompt is
  needed for dattn: a long one puts decode in the sparse regime, which runs the `xq_attn_sel_*`
  path and never reaches `dense_attn_v3`.
- `--probe-exl3-binv` (4.05 pack): `EXL3-ATTN` 300 cells and `EXL3-ATTN-SPARSE` 460 cells pass
  with q4k in the format list (dense v3 == v2, both W4S twins == v2, union == hg3 == hg1 ==
  splitk4).

The attention evidence above is taken on the 4.05 pack.

## 7. Known limitation: the group kernel assumes one slot

`xq_attn_decode_group` (both the v2 and the loop body) resolves one KV slot for the whole group:

```c
const int slot = slotpos[0 * 2 + 0];  // one slot for the whole group
```

That holds for a single request's verify. A packed verify spans two slots, so segment B would
read and write segment A's K/V. q8 has the same latent assumption; it is masked because q8's
packed verify also runs dense-v3, which resolves a slot per row. q4k runs dense-v3 for the same
reason.

The kernel is left untouched: fixing it changes a kernel q8 shares, and the shipped posture
(Flash-Next + default tune + `--lane-pack on`) always takes dense-v3. The unsafe combination is
therefore refused at boot instead: `--lane-pack on` with `--exl3-dense-v3=0` boots with packed
rounds inactive and records the reason in `/metrics velogb10_lane_pack_active{reason=...}`; the F1
boot line that explains it prints on the debug log.

`--exl3-dense-xcheck=1` judges segment B of a packed verify as a mismatch by construction, and
its reference kernel rewrites the same K/V words, so a packed-verify probe run with the flag
fails its own per-segment identity check. Measure packing without it; use the flag on lone
requests.

## 8. Not covered

- **TP=2/4 on hardware**: only the format's transport and the boot-agree hash are exercised
  (single-machine measurements).

## 9. Change log

| date | change |
|---|---|
| 2026-11-02 | First cut: format, readers/writers, dense-v3 and flash256, serving, lane fit and lane packing; measured. |
| 2026-11-02 | Second cut: the QSA gathers (`hg1`/`hg3`, `union`), the pq8 prefill gather and both W4S twins now run q4k; autotune and the docs take the format. |
