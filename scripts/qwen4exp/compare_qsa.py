#!/usr/bin/env python3
# Match the engine's QSA selection lists (stderr of a GB10_QSA_DUMP=1 probe, saved to a file) against
# the HF reference's (ref_sel.json written by ref_forward.py under QSA_DUMP=1). Engine lists are cache
# columns == positions for a single-slot probe; the reference's are token positions. Exact set equality
# is expected except for genuine score ties at the k-th block (reported, not failed).
import sys, json, re
eng_log, ref_json = sys.argv[1], sys.argv[2]
ref = json.load(open(ref_json)); plen, steps = ref['prompt_len'], ref['steps']
# Engine dumps arrive in forward order: the prefill's last query (pos plen-1), then one decode step
# per generated token (pos plen, plen+1, ...); within a step, one dump per attention layer.
nlayers = len({l['layer'] for l in ref['layers']})
# (the probe prefills twice when dumping logits: 'prefill' dumps all describe pos plen-1; 'cols'
# dumps are the decode steps in order, nlayers per step).
eng = {}; escore = {}; ncols = 0; cur_scores = []
for line in open(eng_log):
    m = re.match(r'\[qsa-scores\] (\w+) col=(\d+) scores=\[(.*)\]', line)
    if m: cur_scores.append([float(x) for x in m.group(3).split(',') if x.strip()]); continue
    m = re.match(r'\[qsa-dump\] (\w+) keys=(\S+) col=(\d+) nsel=(\d+) sel=\[(.*)\]', line)
    if not m: continue
    sel = set(int(x) for x in m.group(5).split(',') if x.strip())
    if m.group(1) == 'prefill':
        pos = plen - 1; eng[pos] = []; escore[pos] = []
    else:
        pos = plen + ncols // nlayers; ncols += 1
    eng.setdefault(pos, []).append(sel); escore.setdefault(pos, []).append(cur_scores[-1] if cur_scores else [])
ok = bad = 0
for k in range(steps):
    pos = plen - 1 + k
    if pos not in eng: print(f"pos {pos}: engine dump missing"); bad += 1; continue
    for li, l in enumerate(ref['layers'][:nlayers]):
        r = set(l['sel'][pos]); e = eng[pos][li] if li < len(eng[pos]) else set()
        if r == e: ok += 1
        else:
            bad += 1
            print(f"pos {pos} layer {l['layer']}: ref-only {sorted(r - e)} eng-only {sorted(e - r)} (|ref|={len(r)} |eng|={len(e)})")
            rs = l.get('scores', [[]])[pos]; es = escore.get(pos, [[]])[li] if pos in escore and li < len(escore[pos]) else []
            for blk in sorted({t // 4 for t in (r ^ e)}):
                print(f"    block {blk}: ref score {rs[blk]:.6f}  eng score {es[blk]:.6f}" if blk < len(rs) and blk < len(es) else f"    block {blk}: scores unavailable")
            if rs and es:
                nb = min(len(rs), len(es)); d = [abs(rs[j]-es[j]) for j in range(nb)]
                print(f"    score |diff| over {nb} blocks: max {max(d):.6f} mean {sum(d)/nb:.6f}; ref scores {['%.3f' % x for x in rs]}")
print(f"selection sets identical: {ok}/{ok + bad}")
sys.exit(0 if bad == 0 else 1)
