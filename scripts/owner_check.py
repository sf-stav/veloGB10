#!/usr/bin/env python3
"""What did this server actually do for the traffic it saw? In plain words, from its own counters.

    scripts/owner_check.py --base http://HOST:9000
    scripts/owner_check.py --base http://HOST:9000 --window-seconds 120   # only what happens in the next 2 minutes
    scripts/owner_check.py --file saved_metrics.txt                       # a saved copy of /metrics

Read-only: one or two GETs of /metrics, nothing else. Flags only, no environment.

Why this exists (2026-10-10): a feature was reported "built, gated, x1.2 faster" and did nothing on the
owner's real setup; the server's own counters said so (0 packed rounds) and nobody had looked at them
there. This prints those counters as sentences, so "did it engage for MY requests" takes seconds to
answer and does not depend on anybody's summary.
"""
import argparse
import re
import sys
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# why a request stayed out of a packed round, in plain words (the reasons of src/lane_pack.rs)
SKIP_WORDS = {
    "sampled": "the request uses sampling (temperature above 0) and this boot keeps sampled requests out of packed rounds "
               "(several machines with --lane-pack-sampled-tp off, or an older build)",
    "penalty": "the request carries a repetition / presence / frequency penalty (for example --default-presence-penalty): "
               "penalized requests never join a packed round in this build",
    "few_packable": "it was the only request that could pack at that moment (the others were kept out for another reason)",
    "odd_lane": "an odd number of requests: this one was the odd one out for that round (the role rotates)",
    "prefill": "another request was still reading its prompt (prefill) in that step",
    "mtp_off": "speculation was turned off for this request (it was not paying), so it rode the plain batch",
    "control": "the request owed a control round (the engine re-measures plain speed now and then)",
    "probe": "the request was inside a re-probe window (speculation being tried again after it was turned off)",
    "image": "the request has images",
    "depth0": "no draft fitted in that round (end of the context window or of the token budget)",
    "mixed_regime": "the two requests were on different sides of the long-context attention switch",
    "regime_clamp": "the round was clamped at the long-context attention switch",
    "logprobs_tp": "it asked for logprobs on a TP server (its rows must be full logits rows; no packed graph family keeps them yet)",
    "no_graph": "the packed graph for that shape was not captured at boot",
}


def parse(text):
    out = {}
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        m = LINE.match(raw)
        if not m:
            continue
        labels = tuple(sorted(LABEL.findall(m.group(2) or "")))
        try:
            out[(m.group(1), labels)] = float(m.group(3))
        except ValueError:
            pass
    return out


def fetch(base, timeout):
    url = base.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    with urllib.request.urlopen(url + "/metrics", timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def val(m, name, **labels):
    return m.get((name, tuple(sorted(labels.items()))), 0.0)


def by_label(m, name, label):
    out = {}
    for (n, labels), v in m.items():
        if n == name:
            d = dict(labels)
            if label in d:
                out[d[label]] = out.get(d[label], 0.0) + v
    return out


def diff(a, b):
    """b - a for counters; gauges and build info are taken from b."""
    gauges = ("velogb10_build_info", "velogb10_uptime_seconds", "velogb10_max_batch",
              "velogb10_max_batch_requested", "velogb10_lane_pack_disabled", "velogb10_lane_pack_active",
              "velogb10_exl3_ctx_clamped",
              "velogb10_engine_alive", "velogb10_requests_running", "velogb10_requests_waiting",
              "velogb10_naive_tp_reduce_single_stage", "velogb10_naive_tp_reduce_single_gpu_recv",
              "velogb10_naive_tp_reduce_single_k1_multiblock", "velogb10_naive_moe_shard_tp", "velogb10_naive_lanes",
              "velogb10_fp4l_pack", "velogb10_fp4l_precision",
              "velogb10_copy_span_state", "velogb10_copy_span_min_n", "velogb10_copy_span_max_w",
              "velogb10_copy_span_tp_world", "velogb10_copy_span_yield_gate",
              "velogb10_copy_span_prefill_state",
              "velogb10_copy_span_lanes_state", "velogb10_copy_span_lanes_priced", "velogb10_copy_span_lanes_batch_tok_s",
              "velogb10_copy_span_lanes_copy_round_ms", "velogb10_copy_span_lanes_cursor_accept",
              "velogb10_copy_span_lanes_work_tok_s",
              "velogb10_lane_wide_state", "velogb10_lane_wide_measured_tok_s",
              "velogb10_exl3_dense_resident_state", "velogb10_exl3_dense_resident_matrices", "velogb10_exl3_dense_resident_bytes",
              "velogb10_resolved_default", "velogb10_exl3_mid_gemm_state",
              "velogb10_copy_span_record_state", "velogb10_copy_span_record_read_only",
              "velogb10_copy_span_record_persisting", "velogb10_copy_span_record_ids_held",
              "velogb10_copy_span_record_memory_bytes", "velogb10_copy_span_record_disk_bytes",
              "velogb10_copy_span_record_cap_bytes", "velogb10_copy_span_record_ids_loaded",
              "velogb10_copy_span_record_probe_every", "velogb10_copy_span_record_landing_rate",
              "velogb10_gpu_wait_policy", "velogb10_gpu_wait_context_flags", "velogb10_thermal_guard",
              "velogb10_thermal_celsius", "velogb10_dsv41_markov_dyn_cap", "velogb10_dsv41_copy_draft_min_match")
    return {k: (v if k[0] in gauges else v - a.get(k, 0.0)) for k, v in b.items()}


def n(x):
    return f"{int(round(x)):,}"


def pct(part, whole):
    return f"{100.0 * part / whole:.0f}%" if whole > 0 else "-"


def report(m, where, window):
    L = []
    info = next((dict(labels) for (name, labels) in m if name == "velogb10_build_info"), {})
    build = ", ".join(f"{k} {v}" for k, v in sorted(info.items())) or "build info not exposed"
    up = val(m, "velogb10_uptime_seconds") or val(m, "velogb10_dsv41_uptime_seconds") or val(m, "velogb10_inkling2_uptime_seconds")
    # the Naive-N0.5 server runs its own serve loop: the generic speculation / lane-packing / prime counters below stay 0
    # there and would read as "nothing ran"; its own section (NAIVE-N0.5) reports those instead
    naive = ("velogb10_naive_lanes", ()) in m
    # the DeepSeek-V4.1 serve shell (leg SV) likewise runs its own loop: its own section (DEEPSEEK-V4.1) reports it
    dsv41 = ("velogb10_dsv41_serving", ()) in m
    # the Inkling-2 serve shell (leg T1) likewise: its own section (INKLING-2) reports it
    ink2 = ("velogb10_inkling2_serving", ()) in m
    naive = naive or dsv41 or ink2
    L.append(f"Server {where}")
    up_s = f"up {up / 60:.0f} min" if ("velogb10_uptime_seconds", ()) in m or ("velogb10_dsv41_uptime_seconds", ()) in m or ("velogb10_inkling2_uptime_seconds", ()) in m else "uptime not exposed"
    L.append(f"  {build}; {up_s}; " + (f"DeepSeek-V4.1 server, {int(val(m, 'velogb10_dsv41_lanes'))} request(s) at a time (--max-batch)" if dsv41 else
                                                    f"Inkling-2 server, {int(val(m, 'velogb10_inkling2_lanes'))} request(s) at a time (--max-batch)" if ink2 else
                                                    f"Naive-N0.5 server, {int(val(m, 'velogb10_naive_lanes'))} request(s) at a time (--max-batch)"
                                                    if naive else f"--max-batch {int(val(m, 'velogb10_max_batch'))}"))
    L.append(f"  counted over: {'the last ' + str(window) + ' s' if window else 'everything since the server started'}")
    cap_req, cap = val(m, "velogb10_max_batch_requested"), int(val(m, "velogb10_max_batch"))
    if dsv41:
        # the serve shell reports its own lane rows; the generic max_batch pair is absent (or stale) there, which read as "0 lanes"
        cap_req, cap = val(m, "velogb10_dsv41_lanes_requested"), int(val(m, "velogb10_dsv41_lanes"))
    if ink2:
        cap_req, cap = val(m, "velogb10_inkling2_lanes_requested"), int(val(m, "velogb10_inkling2_lanes"))
    if cap_req > cap:
        L.append(f"  this boot asked for --max-batch {int(cap_req)}: the server serves at most {cap} requests at the same "
                 f"time, so it booted with {cap} lanes and further requests wait in the queue (none is rejected)"
                 + (" (16 is the serving limit)" if cap == 16 else " (16 is the serving limit; memory at this context "
                    "length holds fewer lanes — the boot line says how many)"))
    # CTXCLAMP (owner 2026-11-07): a TP=1 boot whose asked context did not fit even one lane serves a shorter one
    for (name, labels), v in m.items():
        if name == "velogb10_exl3_ctx_clamped" and v > 0:
            d = dict(labels)
            L.append(f"  CONTEXT CLAMPED at boot: --max-seq-len {int(d.get('asked', 0)):,} did not fit even one lane in this box's memory; "
                     f"it serves ONE lane at --max-seq-len {int(d.get('served', 0)):,} (the longest that fits; --tp 2 serves the full context)")
    oc_q = val(m, "velogb10_over_cap_arrivals_total", disposition="queued")
    oc_r = val(m, "velogb10_over_cap_arrivals_total", disposition="rejected")
    if oc_q or oc_r:
        L.append(f"  over-cap arrivals so far: {n(oc_q)} queued (waited for a free lane, then served), "
                 f"{n(oc_r)} rejected with 503 (waiting queue at --max-waiting full)")
    if val(m, "velogb10_engine_alive") == 0 and ("velogb10_engine_alive", ()) in m:
        L.append("  *** THE ENGINE REPORTS ITSELF DEAD (velogb10_engine_alive 0) ***")

    if ink2:
        return L + inkling2_rows(m)
    if dsv41:
        # the serve shell does not feed the generic request / speculation counters below (the generic report would say "nothing was
        # served"): its own rows, plus the Engram and forward blocks the generic report holds, are the whole report
        return L + dsv41_rows(m)
    fin = by_label(m, "velogb10_requests_total", "finish")
    done = sum(fin.values())
    toks = val(m, "velogb10_generation_tokens_total")
    L.append("")
    L.append(f"REQUESTS finished: {n(done)}" + (" (" + ", ".join(f"{k} {n(v)}" for k, v in sorted(fin.items()) if v) + ")" if done else "")
             + f"; tokens generated: {n(toks)}")
    if done == 0 and toks == 0:
        L.append("  Nothing was served in this period, so there is nothing to judge. Send your requests, then run this again.")
        return L
    tba = val(m, "velogb10_thinking_budget_requests_total", disposition="applied")
    tbu = val(m, "velogb10_thinking_budget_requests_total", disposition="unsupported")
    if tba or tbu:
        L.append(f"  thinking budget: {n(tba)} request(s) sent thinking_token_budget and had it applied; it ended the reasoning "
                 f"{n(val(m, 'velogb10_thinking_budget_forced_total'))} time(s)"
                 + (f"; {n(tbu)} request(s) ran WITHOUT it (this model or serving path cannot apply it)" if tbu else ""))
    # PR #30 / #31 ports: log probabilities and json_schema enforcement, and what they cost in speculation
    lps = {d: val(m, "velogb10_logprobs_requests_total", disposition=d) for d in ("served", "unsupported", "withheld", "stream")}
    if any(lps.values()):
        L.append(f"  logprobs: {n(lps['served'])} request(s) got them"
                 + "".join(f"; {n(lps[d])} {what}" for d, what in (
                     ("unsupported", "got logprobs null (this serving engine computes none: the EXL3 server at TP=1, 2 and 4 does)"),
                     ("withheld", "got logprobs null (--scrub-tool-secrets is on)"),
                     ("stream", "got none (stream=true is not wired for them yet)")) if lps[d])
                 + f"; {n(val(m, 'velogb10_logprobs_spec_rows_total'))} token(s) took theirs from speculative rounds (logprobs keep speculation)")
    se = val(m, "velogb10_json_schema_requests_total", disposition="enforced")
    su = val(m, "velogb10_json_schema_requests_total", disposition="unenforced")
    if se or su:
        L.append(f"  response_format json_schema: {n(se)} request(s) enforced by a token mask (x-json-schema-enforced: full), "
                 f"{n(su)} served UNCONSTRAINED (x-json-schema-enforced: none: --json-schema-enforce off, a thinking request "
                 f"on an engine that cannot defer the mask, the EXL3 TP head, or an unsupported schema)")
    sol = val(m, "velogb10_spec_off_lanes_total", why="logprobs")
    sos = val(m, "velogb10_spec_off_lanes_total", why="schema")
    if sol or sos:
        L.append(f"  *** {n(sol + sos)} EXL3 request(s) decoded WITHOUT speculation (plain steps, roughly half the speed): "
                 f"{n(sol)} asked for logprobs (older builds), {n(sos)} for a schema mask (its mask does not cover the verify rows yet) ***")
    tk = val(m, "velogb10_top_k_capped_total")
    if tk:
        L.append(f"  top_k capped: {n(tk)} sampled request(s) asked for no top-k truncation (top_k 0 or negative) or more than 64 "
                 f"candidates; this model family's GPU sampler keeps at most 64, so they were served with top_k 64 "
                 f"(the EXL3 server and --cpu-sample apply no such limit)")
    dc, ds = val(m, "velogb10_decode_tokens_per_second_count"), val(m, "velogb10_decode_tokens_per_second_sum")
    # LANERES2: the SSD spill tier for evicted lane snapshots (--kv-spill).
    sw, sh = val(m, "velogb10_kv_spill_writes_total"), val(m, "velogb10_kv_spill_hits_total")
    sm = val(m, "velogb10_kv_spill_misses_recompute_total")
    sb, stv = val(m, "velogb10_kv_spill_bytes"), val(m, "velogb10_kv_spill_tokens_recompute_saved_total")
    if sw or sh or sm:
        L.append(f"  kv spill: {int(sw)} snapshots written ({sb / 2**30:.2f} GiB), {int(sh)} restores "
                 f"(saved {int(stv)} prompt tokens of re-prefill), {int(sm)} evictions re-prefilled")
    else:
        L.append("  kv spill: idle (no evictions captured yet, or the tier is off)")
    # PFXPOOL: the prefix pool (prompt states kept in memory across lanes, --prefix-pool-gb).
    pcap, pby, pen = val(m, "velogb10_prefix_pool_cap_bytes"), val(m, "velogb10_prefix_pool_bytes"), val(m, "velogb10_prefix_pool_entries")
    pk, ph, pht = val(m, "velogb10_prefix_pool_captures_total"), val(m, "velogb10_prefix_pool_hits_total"), val(m, "velogb10_prefix_pool_hit_tokens_total")
    pr, pe, pem = val(m, "velogb10_prefix_pool_refused_total"), val(m, "velogb10_prefix_pool_evictions_total"), val(m, "velogb10_prefix_pool_emergency_frees_total")
    if pcap:
        L.append(f"  prefix pool: holds {int(pen)} prompts ({pby / 2**30:.2f} of {pcap / 2**30:.0f} GiB); {int(pk)} kept, {int(ph)} prompts "
                 f"resumed from it ({n(pht)} prompt tokens not read again); {int(pe)} shifted out, {int(pr)} not kept "
                 f"(over the cap / memory floor / failed copy), {int(pem)} freed early because a box went under the memory floor")
    else:
        L.append("  prefix pool: off (--prefix-pool-gb 0 or --prefix-cache off)")
    # KVPOOL (veloGB10 #43, @AbsolomBane): the shared KV page pool behind the EXL3 position planes (--kv-pool).
    kvm = val(m, "velogb10_kv_vmm_mode")
    kfb_tp, kfb_model = val(m, "velogb10_kv_vmm_fallback", reason="tp"), val(m, "velogb10_kv_vmm_fallback", reason="model")
    if kvm:
        kpool, kmap = val(m, "velogb10_kv_vmm_pool_bytes"), val(m, "velogb10_kv_vmm_mapped_bytes")
        kev, kle, kw = val(m, "velogb10_kv_vmm_evictions_total"), val(m, "velogb10_kv_vmm_length_ends_total"), val(m, "velogb10_kv_vmm_waits_total")
        L.append(f"  shared KV pool: ON ({'lazy: pages mapped on demand' if kvm == 2 else 'full: every page mapped at boot'}); "
                 f"{kmap / 2**30:.2f} of {kpool / 2**30:.2f} GiB in use; pool full so far: {int(kev)} idle lanes evicted, "
                 f"{int(kle)} requests ended early (finish=length, the pool could not grow them), {int(kw)} prompts waited for room")
    elif kfb_tp or kfb_model:
        L.append("  shared KV pool: asked for (--kv-pool) but NOT running here: "
                 + ("a TP rank serves the standard per-lane KV allocation" if kfb_tp else "this model's KV is not in the EXL3 position planes the pool backs"))
    elif ("velogb10_kv_vmm_mode", ()) in m:
        L.append("  shared KV pool: off (the standard per-lane KV allocation; --kv-pool lazy turns it on)")
    # VISCACHE (veloGB10 #42, @rusty-dev): vision-tower outputs kept keyed on the image content (--vision-embed-cache-mb).
    vh, vmiss = val(m, "velogb10_vision_cache_hits_total"), val(m, "velogb10_vision_cache_misses_total")
    vb, vev = val(m, "velogb10_vision_cache_bytes"), val(m, "velogb10_vision_cache_evictions_total")
    if vh or vmiss:
        L.append(f"  vision embed cache: {int(vh)} images served from the cache (the vision tower did not run for them), "
                 f"{int(vmiss)} ran the tower ({pct(vh, vh + vmiss)} hits); {vb / 2**20:.1f} MiB held, {int(vev)} shifted out")
    elif ("velogb10_vision_cache_hits_total", ()) in m:
        L.append("  vision embed cache: no image requests yet (or --vision-embed-cache-mb 0)")
    # TURNRESUME: --turn-resume / --next-turn-prefill (the next turn resumes from a state that already holds the reply).
    tmr, tmn = val(m, "velogb10_turn_mode_resume"), val(m, "velogb10_turn_mode_next_turn_prefill")
    if tmr == 1:
        th, tht = val(m, "velogb10_turn_hits_total"), val(m, "velogb10_turn_hit_tokens_total")
        tmd = val(m, "velogb10_turn_misses_total", reason="diverged")
        tmnone, tmnd = val(m, "velogb10_turn_misses_total", reason="none"), val(m, "velogb10_turn_misses_total", reason="not_deeper")
        tf, tfp = val(m, "velogb10_turn_followups_total"), val(m, "velogb10_turn_followup_prefilled_tokens_total")
        bp, ba, bt = val(m, "velogb10_turn_bg_passes_total"), val(m, "velogb10_turn_bg_aborted_total"), val(m, "velogb10_turn_bg_tokens_total")
        bw, bwu, bwm = val(m, "velogb10_turn_bg_waits_total"), val(m, "velogb10_turn_bg_wait_us_total"), val(m, "velogb10_turn_bg_wait_max_us")
        hp, hn, hm = val(m, "velogb10_turn_hints_posted_total"), val(m, "velogb10_turn_hints_nolane_total"), val(m, "velogb10_turn_hints_mismatch_total")
        bf = val(m, "velogb10_turn_bg_failed_total")
        L.append(f"  turn resume: ON ({val(m, 'velogb10_turn_state_bytes') / 2**30:.2f} GiB of kept states); {int(th)} follow-ups resumed "
                 f"from a state that already held the previous reply ({n(tht)} tokens they did not read again); not used: "
                 f"{int(tmd)} sent something else than the kept reply (edited, branched), {int(tmnone)} had no kept state, "
                 f"{int(tmnd)} not deeper ({int(val(m, 'velogb10_turn_misses_total', reason='other_conversation'))} lanes taken by "
                 f"another conversation). {int(tf)} follow-ups in all still read {n(tfp)} prompt tokens "
                 f"({tfp / tf:.0f} each)" if tf else
                 f"  turn resume: ON, no follow-up request yet")
        if tmn == 1:
            L.append(f"  next-turn prefill: {int(hp)} replies rendered for the next turn ({int(hn)} with no idle lane left, {int(hm)} that "
                     f"did not start with the prompt), {int(bp)} idle-time passes read {n(bt)} tokens; {int(ba)} yielded to an arriving "
                     f"request, which waited {bwu / max(bw, 1) / 1e3:.0f} ms on average (longest {bwm / 1e3:.0f} ms) for the chunk in flight"
                     + (f"; {int(bf)} passes FAILED" if bf else ""))
    elif tmr == 2:
        L.append("  turn resume: asked for but running OFF (see the boot line: no memory / TP / no prefix cache)")
    # PERFFIX1: --prefill-interleave (decode steps between the chunks of a joining prompt; every topology).
    ist, ipt = val(m, "velogb10_prefill_interleave_steps_total"), val(m, "velogb10_prefill_interleave_points_total")
    if ist:
        L.append(f"  prefill interleave: {int(ist)} decode steps ran between the chunks of joining prompts, at {int(ipt)} chunk "
                 "boundaries (running streams kept moving while a long prompt was read)")
    # PACKPF: --prefill-pack (several waiting prompts prefilled in shared passes; TP=1).
    pp, ppk, pps = val(m, "velogb10_prefill_pack_passes_total"), val(m, "velogb10_prefill_pack_packed_passes_total"), val(m, "velogb10_prefill_pack_segments_total")
    ppr, ppn, pph = val(m, "velogb10_prefill_pack_rows_total"), val(m, "velogb10_prefill_pack_prompts_total"), val(m, "velogb10_prefill_pack_holds_total")
    ppskip = {k: v for k, v in by_label(m, "velogb10_prefill_pack_skipped_total", "reason").items() if v}
    if pp or ppskip:
        L.append(f"  packed prefill: {int(ppn)} prompts prefilled by the packed admission in {int(pp)} passes, {int(ppk)} of them "
                 f"shared by 2+ prompts ({int(pps)} prompt segments, {n(ppr)} rows); {int(pph)} admission holds for a fuller pass"
                 + (" -- NOT PACKED: " + ", ".join(f"{k} {int(v)}" for k, v in sorted(ppskip.items())) if ppskip else ""))
    # LANERES3: the TP tier of --lane-reserve / --kv-spill (head-authoritative admits).
    rq = val(m, "velogb10_lane_reserve_tp_requeued_total")
    pd, pm = val(m, "velogb10_kv_spill_capture_peer_drop_total"), val(m, "velogb10_kv_spill_restore_peer_miss_total")
    if rq or pd or pm:
        L.append(f"  lane reserve TP: {int(rq)} admits re-planned onto a lane both ranks agree free "
                 f"(never a refusal); spill cross-rank guard: {int(pd)} captures dropped, {int(pm)} restores "
                 f"dropped + re-prefilled (a rank could not read or parse the shard, so nobody restores)")
    if dc > 0:
        L.append(f"  average decode speed of a request: {ds / dc:.1f} tok/s over {n(dc)} requests "
                 f"(each request's own speed; with several at once the total is higher than this)")

    # V0710: the family/TP-scoped defaults this boot resolved (auto = the server chose, set = given explicitly)
    rd = [dict(labels) for (name, labels) in m if name == "velogb10_resolved_default"]
    if rd:
        L.append("")
        L.append("RESOLVED DEFAULTS (copy-span, its 16-row copy check and dense-resident are on by default for the "
                 "Flash-Next family only; mid-GEMM is opt-in; the arm meter is v2 at TP=1/2 and v1 at TP=4; anything typed on the command line wins)")
        for d in sorted(rd, key=lambda d: d.get("option", "")):
            L.append(f"  --{d.get('option')} {d.get('value')} ({'chosen by the server' if d.get('source') == 'auto' else 'given explicitly'})")
        if ("velogb10_exl3_mid_gemm_state", ()) in m:
            if val(m, "velogb10_exl3_mid_gemm_state") > 0:
                L.append(f"  mid-GEMM (--exl3-mid-gemm) ON: {n(val(m, 'velogb10_exl3_mid_gemm_chains_total'))} dense chains of 17-64 rows "
                         f"({n(val(m, 'velogb10_exl3_mid_gemm_rows_total'))} rows) read the packed weights directly"
                         + ("" if val(m, "velogb10_exl3_mid_gemm_chains_total") > 0 else " - NONE yet"))
            else:
                L.append("  mid-GEMM (--exl3-mid-gemm) OFF on this boot")

    L += heat_rows(m)

    pt, pc = val(m, "velogb10_prompt_tokens_total"), val(m, "velogb10_prompt_tokens_cached_total")
    if pt > 0:
        L.append("")
        L.append("PROMPTS")
        L.append(f"  prompt tokens received: {n(pt)}; taken from the prompt cache instead of being read again: {pct(pc, pt)} ({n(pc)})")
        if naive:
            L.append("  (the Naive prefix reuse keeps one conversation per KV slot (--max-batch): a prompt that continues none of them reads its whole prompt)")
        else:
            L.append("  (the cache is per lane: a conversation whose lane is busy or was reused by another one reads its whole prompt again,")
            L.append("   and every running stream stands still while a prompt is read)")
    ge = val(m, "velogb10_graph_cache_entries")
    if ("velogb10_graph_cache_entries", ()) in m and not naive:
        L.append("")
        L.append("GPU GRAPHS (pre-recorded launch sequences; a shape without one runs the slow way the first time)")
        L.append(f"  graphs recorded during this period: {n(ge)}" if window else
                 f"  graphs held now: {n(ge)} (boot records the greedy, unpenalized shapes; sampled and penalized ones are recorded during traffic)")

    if naive:
        L.append("")
        L.append("SPECULATION AND CONCURRENCY: this is a Naive-N0.5 server; see its own section (NAIVE-N0.5) at the end")
    else:
        rounds, emitted = val(m, "velogb10_spec_rounds_total"), val(m, "velogb10_spec_emitted_tokens_total")
        draft, acc = val(m, "velogb10_spec_draft_tokens_total"), val(m, "velogb10_spec_accepted_tokens_total")
        L.append("")
        L.append("SPECULATION (the engine drafts several tokens ahead and keeps the ones that check out; requests still running may not be counted yet)")
        if rounds > 0:
            L.append(f"  speculative rounds: {n(rounds)}; tokens produced per round: {emitted / rounds:.2f} "
                     f"(1.00 would mean speculation gains nothing)")
            L.append(f"  drafted tokens that were accepted: {pct(acc, draft)} ({n(acc)} of {n(draft)})")
            if toks > 0:
                L.append(f"  share of all generated tokens that came out of speculative rounds: {pct(emitted, toks)}")
        else:
            L.append("  NO speculative round ran in this period (speculation off, or every step was a shared plain step).")
        off = val(m, "velogb10_spec_mtp_off_events_total")
        if ("velogb10_spec_mtp_off_events_total", ()) not in m:
            L.append("  speculation switch-off events: NOT EXPORTED by this build (no statement about them)")
        if off > 0:
            L.append(f"  the engine turned speculation OFF for a request {n(off)} time(s) because it was not paying there; "
                     f"tried it again {n(val(m, 'velogb10_spec_mtp_reprobes_total'))} time(s), "
                     f"{n(val(m, 'velogb10_spec_mtp_reprobe_ok_total'))} of them successful")
        # --draft-vocab (FR-Spec, opt-in): the MTP drafter proposes from a reduced vocabulary.
        if ("velogb10_draft_vocab_rows", ()) in m:
            dv_rows, dv_full = val(m, "velogb10_draft_vocab_rows"), val(m, "velogb10_draft_vocab_full_rows")
            dv_steps = val(m, "velogb10_draft_vocab_steps_total")
            if dv_rows > 0:
                L.append(f"  reduced draft vocabulary (--draft-vocab): ON, the drafter proposes from {n(dv_rows)} of "
                         f"{n(dv_full)} tokens ({pct(dv_rows, dv_full)}); draft tokens it picked: {n(dv_steps)}")
            else:
                L.append("  reduced draft vocabulary (--draft-vocab): off, the drafter reads the full vocabulary")

        # ISSUES[94]: DFlash2 (DF2) chain speculation on the EXL3 server (TP=1). The boot line
        # `[exl3-serve] df2: ENGAGED ...` / `FALLBACK to pack MTP (<reason>)` says why it engaged
        # or not; these counters say what it actually did afterwards (a boot that engaged but never
        # served a round is visible here, not silent).
        if ("exl3_df2_engaged", ()) in m:
            eng = val(m, "exl3_df2_engaged")
            df2_rounds = val(m, "exl3_df2_rounds_total")
            df2_acc = val(m, "exl3_df2_tokens_accepted_total")
            df2_fb = val(m, "exl3_df2_fallback_mtp_total")
            df2_dds = val(m, "exl3_df2_dds_bypass_total")
            df2_ship = val(m, "exl3_df2_shipped_drafts_total")
            L.append("")
            L.append("DFLASH2 CHAIN (the --spec-source dflash2* / --draft-dir speculator on the EXL3 server; "
                     "it drafts with the DF2 chain instead of the pack's MTP head)")
            if eng >= 0.5:
                L.append("  ENGAGED on this boot")
                if df2_rounds > 0:
                    L.append(f"  DFlash2 rounds: {n(df2_rounds)}; accepted tokens per round: {df2_acc / df2_rounds:.2f} "
                             f"(1.00 would mean the chain gains nothing; higher is faster)")
                else:
                    L.append("  NO DFlash2 round served yet: nothing generated, or every solo round went elsewhere "
                             "(see the bypass / fallback lines right below)")
                if df2_dds > 0:
                    L.append(f"  {n(df2_dds)} solo round(s) were served by the adaptive/DDS path instead of the chain "
                             f"(a --draft-confidence was set: the DDS sampler cannot ride the DF2 chain, so it kept its own path)")
                if df2_ship > 0:
                    L.append(f"  {n(df2_ship)} sampled DF2 round(s) used the head's draft on every machine (shipped per round; "
                             "the fix for the TP rank divergence, ISSUES 117)")
                if df2_fb > 0:
                    L.append(f"  {n(df2_fb)} round(s) fell back to the pack MTP head (that lane's context window was not "
                             f"primed for the chain, e.g. right after a lane switch; served, not refused)")
            else:
                L.append("  NOT ENGAGED on this boot — the server's boot log says why: look for the "
                         "`[exl3-serve] df2: FALLBACK to pack MTP (<reason>)` line (reason is one of: source, no --draft-dir, "
                         "dims mismatch, ctx longer than 262144, or load failed: <error>)")
        else:
            L.append("")
            L.append("DFLASH2 CHAIN: this build does not export the exl3_df2_* counters (a build without the DF2 serve path, "
                     "or the /metrics render hook for them is not in yet) — no statement about the chain; "
                     "the boot log line `[exl3-serve] df2: ...` is the fallback evidence")

        # WIDEVERIFY1: copy-span drafting (--copy-span on): a lone request's speculative round drafts
        # tokens copied from its own context (no draft pass). Its rounds are ALSO inside the
        # speculative-round totals above.
        if ("velogb10_copy_span_state", ()) in m:
            cst = val(m, "velogb10_copy_span_state")
            L.append("")
            L.append("COPY-SPAN DRAFTING (--copy-span: when the answer copies text from its own prompt or earlier output, "
                     "a speculative round drafts that text instead of asking the draft head)")
            if cst < 0.5:
                L.append("  not asked for on this boot (--copy-span off, the default)")
            elif cst < 1.5:
                cr = val(m, "velogb10_copy_span_rounds_total")
                co, ca = val(m, "velogb10_copy_span_offered_tokens_total"), val(m, "velogb10_copy_span_accepted_tokens_total")
                ce, cz = val(m, "velogb10_copy_span_emitted_tokens_total"), val(m, "velogb10_copy_span_zero_accept_rounds_total")
                csec = val(m, "velogb10_copy_span_round_seconds_total")
                lanes_on = ("velogb10_copy_span_lanes_state", ()) in m and 0.5 < val(m, "velogb10_copy_span_lanes_state") < 1.5
                trunk = ("velogb10_copy_span_trunk", ()) in m and val(m, "velogb10_copy_span_trunk") > 0.5
                L.append(f"  ON on this boot (a copy starts on a match of {n(val(m, 'velogb10_copy_span_min_n'))} tokens, "
                         f"at most {n(val(m, 'velogb10_copy_span_max_w'))} copied tokens per round; "
                         + ("also while 2 to 4 requests are busy: see COPY-SPAN FOR CONCURRENT REQUESTS below)" if lanes_on else
                            "one request at a time only: 2 or more concurrent requests keep the batched paths)" if trunk else
                            "one request at a time only: 2 to 4 concurrent requests use the draft head as before)"))
                tw = val(m, "velogb10_copy_span_tp_world")
                if tw > 1.5:
                    L.append(f"  TP={n(tw)}: every machine runs the same copy proposer; copy rounds whose copied tokens all machines "
                             f"agreed on before running: {n(val(m, 'velogb10_copy_span_tp_agreed_total'))}")
                if cr > 0:
                    L.append(f"  copy rounds: {n(cr)} of {n(rounds)} speculative rounds ({pct(cr, rounds)}); they produced "
                             f"{n(ce)} tokens ({ce / cr:.2f} per round; {pct(ce, emitted)} of the speculative tokens)")
                    L.append(f"  copied tokens the model kept: {pct(ca, co)} ({n(ca)} of {n(co)}); "
                             f"copy rounds where the first copied token was wrong: {n(cz)}")
                    if csec > 0:
                        L.append(f"  average copy round: {1000.0 * csec / cr:.1f} ms")
                else:
                    L.append("  NO copy round ran in this period (no request copied its context, or only concurrent requests ran)")
                ps, pc = val(m, "velogb10_copy_span_proposals_total", source="search"), val(m, "velogb10_copy_span_proposals_total", source="cursor")
                mn, mb = val(m, "velogb10_copy_span_misses_total", reason="no_match"), val(m, "velogb10_copy_span_misses_total", reason="below_min")
                L.append(f"  copies started by a context match: {n(ps)}; rounds that continued a running copy: {n(pc)}")
                L.append(f"  rounds with no copy proposal: {n(mn + mb)} ({n(mn)} found no earlier occurrence of the last tokens, "
                         f"{n(mb)} found one shorter than the minimum match) - those rounds used the draft head")
                em = val(m, "velogb10_copy_span_copy_ends_total", reason="mismatch")
                eu = val(m, "velogb10_copy_span_copy_ends_total", reason="unused")
                L.append(f"  copies that ended because the model wrote something else: {n(em)}; "
                         f"interrupted by a round of another kind (a control round or a shared step): {n(eu)}")
                if ("velogb10_copy_span_start_w", ()) in m:
                    sw, mw = val(m, "velogb10_copy_span_start_w"), val(m, "velogb10_copy_span_max_w")
                    ns = val(m, "velogb10_copy_span_narrow_proposals_total", source="search")
                    nc = val(m, "velogb10_copy_span_narrow_proposals_total", source="cursor")
                    if 0.5 < sw < mw:
                        L.append(f"  start ramp ON (--copy-span-start-w {n(sw)}): a copy's first round proposes at most {n(sw)} tokens "
                                 f"and each following round doubles that, so a copy the model rejects at once pays for a short check; "
                                 f"copies that started narrow: {n(ns)}; later rounds still widening: {n(nc)}")
                    else:
                        L.append(f"  start ramp OFF (--copy-span-start-w {n(sw)} >= the width): every copy round proposes the full width")
                if trunk:
                    L.append(f"  trunk server (NVFP4/bf16 pack): a copy round replaces that round of the DFlash2 chain or the MTP head "
                             f"and checks up to {n(min(val(m, 'velogb10_copy_span_max_w'), 15))} copied tokens in one verify (16 rows at most)")
                    L.append("  with the copy arm on, the MTP head's depth policy opens at the deepest depth, re-evaluates every 32 "
                             "draft-head rounds (not 128) and runs every 8th window at the deepest depth (boot-wide; the server log's "
                             "[mtp] lines show it; a pinned --mtp-depth is not changed); per-request [copy-acct] log lines split "
                             "rounds / tokens / ms by reasoning vs answer and by round kind")
                    if ("velogb10_copy_span_bare_rounds_total", ()) in m:
                        L.append(f"  copy rounds served to a lone request whose draft head was not live (after 2+ requests ran at once, "
                                 f"or speculation off), in place of one-token plain steps: {n(val(m, 'velogb10_copy_span_bare_rounds_total'))}")
                    sk = {r: val(m, "velogb10_copy_span_skipped_rounds_total", reason=r)
                          for r in ("concurrent", "schema", "source", "not_primed", "spec_off", "tp_disagree")}
                    L.append(f"  steps that could NOT take a copy round (they ran as without the flag): {n(sum(sk.values()))} - "
                             f"{n(sk['concurrent'])} with 2+ requests at once, {n(sk['schema'])} JSON-schema requests, "
                             f"{n(sk['source'])} on a drafter without copy wiring, {n(sk['not_primed'])} whose drafter state was not live "
                             f"(plain decode), {n(sk['spec_off'])} with speculation off, {n(sk['tp_disagree'])} where the TP machines proposed "
                             f"different copies (all ran the draft head)")
                if not trunk and val(m, "velogb10_copy_span_max_w") > 10.5:
                    L.append(f"  WIDE copy check (--copy-span-max-w {n(val(m, 'velogb10_copy_span_max_w'))} > 10): copy rounds check up "
                             f"to 15 copied tokens at once (the draft head's rounds keep 10); not yet proven on hardware")
                if ("velogb10_copy_span_yield_gate", ()) in m:
                    if val(m, "velogb10_copy_span_yield_gate") > 0.5:
                        yd = val(m, "velogb10_copy_span_yield_gate_denied_total")
                        yp = val(m, "velogb10_copy_span_yield_gate_probes_total")
                        L.append(f"  yield gate ON (--copy-span-yield-gate): copies NOT started because copying was not paying on "
                                 f"that request (its copy rounds made fewer tokens than its draft-head rounds were worth): {n(yd)} - "
                                 f"the draft head served those rounds; held-back copies started anyway as a periodic re-check: {n(yp)}")
                    else:
                        L.append("  yield gate OFF (--copy-span-yield-gate off): every match starts a copy, whether or not copying pays")
            else:
                why = {2: (f"this TP world size (TP={n(val(m, 'velogb10_copy_span_tp_world'))}) is not wired: the copy arm serves "
                           + "one box, TP=2 and TP=4"),
                       3: "this pack's server has no copy arm (DeepSeek V4 / Naive bundles; the EXL3 and NVFP4/bf16 servers have one)",
                       4: "speculation itself is off on this boot (--mtp off or no draft head)"}.get(int(round(cst)), "unknown reason")
                L.append(f"  asked for (--copy-span on) but running OFF: {why}; the server runs as without the flag")

        # COPYLANES: --copy-span-lanes: copy work for every busy request (2-4 at once), priced against the batch.
        if ("velogb10_copy_span_lanes_state", ()) in m:
            lst = val(m, "velogb10_copy_span_lanes_state")
            L.append("")
            L.append("COPY-SPAN FOR CONCURRENT REQUESTS (--copy-span-lanes: while 2 to 4 requests are busy, a request that is copying "
                     "its context runs its own copy rounds / stretches in the step; the others run the step as before)")
            if lst < 0.5:
                L.append("  not asked for on this boot (--copy-span-lanes off, the default): copy work serves a lone request only")
            elif lst < 1.5:
                wr = val(m, "velogb10_copy_span_lanes_work_total", kind="round")
                ws = val(m, "velogb10_copy_span_lanes_work_total", kind="stretch")
                wo = val(m, "velogb10_copy_span_lanes_work_total", kind="other")
                we, wp = val(m, "velogb10_copy_span_lanes_emitted_tokens_total"), val(m, "velogb10_copy_span_lanes_probes_total")
                st = val(m, "velogb10_copy_span_lanes_steps_total")
                L.append("  ON on this boot, " + ("priced: a request leaves the step for copy work only when it delivers at least 1.15x "
                                                   "the batch's measured tokens/s" if val(m, "velogb10_copy_span_lanes_priced") > 0.5
                                                   else "price OFF (diagnostic): every copy proposal runs"))
                if wr + ws + wo > 0:
                    L.append(f"  busy steps with copy work: {n(st)}; copy rounds {n(wr)}, stretches {n(ws)}, taken but served "
                             f"another round (a control round had priority): {n(wo)}; tokens they produced: {n(we)}"
                             + (f" ({we / (wr + ws):.1f} per copy round or stretch)" if wr + ws > 0 else "")
                             + f"; taken unpriced to measure the costs: {n(wp)}")
                else:
                    L.append("  NO copy work ran while other requests were busy in this period")
                why = {"no_copy": "nothing to copy", "priced_out": "copy work that would not beat the batch",
                       "spec_off": "request does not speculate", "clamped": "round depth 0 (window / budget)"}
                held = [(r, val(m, "velogb10_copy_span_lanes_skip_total", reason=r)) for r in why]
                L.append("  busy requests kept in the step (per request per step): "
                         + "; ".join(f"{why[r]}: {n(v)}" for r, v in held))
                sp = val(m, "velogb10_copy_span_lanes_stretch_priced_out_total")
                if sp > 0:
                    L.append(f"  stretches held back in busy steps (a copy round served instead): {n(sp)}")
                L.append(f"  last price: batch {val(m, 'velogb10_copy_span_lanes_batch_tok_s'):.0f} tok/s; copy round "
                         f"{val(m, 'velogb10_copy_span_lanes_copy_round_ms', width='chain'):.1f} ms (chain width) / "
                         f"{val(m, 'velogb10_copy_span_lanes_copy_round_ms', width='wide'):.1f} ms (wide); running-copy accept "
                         f"{100 * val(m, 'velogb10_copy_span_lanes_cursor_accept'):.0f}%; copy work {val(m, 'velogb10_copy_span_lanes_work_tok_s'):.0f} tok/s")
            else:
                why = {2: "--copy-span is not armed on this boot", 3: "--lane-order fcfs steps one request at a time",
                       4: "--max-batch 1 never has two busy requests",
                       5: "this server has no busy-step copy work (built into the EXL3 server only; the NVFP4/bf16 server's "
                          "copy arm serves a lone request, DeepSeek V4 / Naive bundles have no copy arm)"}.get(int(round(lst)), "unknown reason")
                L.append(f"  asked for (--copy-span-lanes on) but running OFF: {why}; the server runs as without the flag")

        # COPYPF1: --copy-span-prefill: an established copy's next stretch is checked by ONE prompt-reading pass
        # (a chunked prefill) instead of copy rounds. Its stretches are ALSO inside the speculative-round totals.
        if ("velogb10_copy_span_prefill_state", ()) in m:
            pst = val(m, "velogb10_copy_span_prefill_state")
            L.append("")
            L.append("COPY-SPAN PREFILL CHECK (--copy-span-prefill: once a copy is established, the next stretch of copied text "
                     "is checked in one prompt-reading pass; OUTPUT MAY DIFFER ON NEAR-TIES from the default path)")
            if pst < 0.5:
                L.append("  not asked for on this boot (--copy-span-prefill off, the default)")
            elif pst < 1.5:
                ps = val(m, "velogb10_copy_span_prefill_steps_total")
                po, pa = val(m, "velogb10_copy_span_prefill_offered_tokens_total"), val(m, "velogb10_copy_span_prefill_accepted_tokens_total")
                pe, pd = val(m, "velogb10_copy_span_prefill_emitted_tokens_total"), val(m, "velogb10_copy_span_prefill_disagree_steps_total")
                pr, pj = val(m, "velogb10_copy_span_prefill_rerun_rows_total"), val(m, "velogb10_copy_span_prefill_rows_judged_total")
                psec = val(m, "velogb10_copy_span_prefill_seconds_total")
                L.append("  ON on this boot (one request at a time, TP=1, 2 and 4; not for penalized, logprob or image requests)")
                if ps > 0:
                    L.append(f"  stretches checked: {n(ps)}; they produced {n(pe)} tokens ({pe / ps:.1f} per stretch"
                             + (f", {pe / psec:.0f} tokens/s while checking" if psec > 0 else "") + ")")
                    L.append(f"  copied tokens the model kept: {pct(pa, po)} ({n(pa)} of {n(po)}); stretches that stopped where the "
                             f"model chose another token: {n(pd)}; rows read a second time after such a stop: {n(pr)}; "
                             f"rows checked by the model: {n(pj)}")
                else:
                    L.append("  NO stretch was checked this way in this period (no established copy, or only concurrent requests ran)")
                why = {"penalty": "penalized request", "logprobs": "log probabilities asked", "image": "image request",
                       "prefill_interleave": "a prompt was being read between steps", "short": "fewer than 16 copied tokens left",
                       "no_tap": "no draft-head tap saved", "no_scratch": "no prompt-reading buffers yet",
                       "young_copy": "the copy was younger than --copy-span-prefill-min-run",
                       "priced_out": "priced out (the copy was not expected to run long enough to beat copy rounds)"}
                held = [(r, val(m, "velogb10_copy_span_prefill_fallback_total", reason=r)) for r in why]
                held = [(r, v) for r, v in held if v > 0]
                if held:
                    L.append("  established copies that used copy rounds instead: "
                             + "; ".join(f"{why[r]}: {n(v)}" for r, v in held))
                # ECHOPRICE: the priced rule (--copy-span-prefill-price, default on) and the costs it compares
                prule = val(m, "velogb10_copy_span_prefill_price_rule")
                cms, ctok = val(m, "velogb10_copy_span_prefill_price_copy_round_ms"), val(m, "velogb10_copy_span_prefill_price_copy_round_tokens")
                pms = [val(m, "velogb10_copy_span_prefill_price_stretch_ms", rung=str(i)) for i in range(4)]
                if prule:
                    L.append("  priced rule ON (--copy-span-prefill-price): a stretch is taken only when the copy is expected to run long "
                             "enough to beat copy rounds; measured: full copy round "
                             + (f"{cms:.1f} ms for {ctok:.0f} tokens" if cms > 0 else "not yet")
                             + "; stretch per rung " + ", ".join(f"{x:.0f} ms" if x > 0 else "-" for x in pms))
                else:
                    L.append("  priced rule OFF (--copy-span-prefill-price off): every established copy takes a stretch from the first rung")
            else:
                why = {2: "this is a TP server (this version is wired for one box only)",
                       3: "--copy-span is not armed on this boot (the check verifies copies made by --copy-span)",
                       4: "the DFlash2 chain is armed on this boot (not wired for stretches)"}.get(int(round(pst)), "unknown reason")
                L.append(f"  asked for (--copy-span-prefill on) but running OFF: {why}; copy rounds are unchanged")

        # LANEWIDE: --lane-wide-verify: several busy requests' speculative rows verified by ONE prefill-class pass.
        if ("velogb10_lane_wide_state", ()) in m:
            wst = val(m, "velogb10_lane_wide_state")
            L.append("")
            L.append("LANE-WIDE VERIFY (--lane-wide-verify: with several busy requests, ONE prefill-class pass checks every request's "
                     "drafts; OUTPUT MAY DIFFER ON NEAR-TIES from the default path)")
            if wst < 0.5:
                L.append("  not asked for on this boot (--lane-wide-verify off, the default)")
            elif wst < 2.5:
                wr, wl = val(m, "velogb10_lane_wide_rounds_total"), val(m, "velogb10_lane_wide_lane_rounds_total")
                wt, wrows = val(m, "velogb10_lane_wide_tokens_total"), val(m, "velogb10_lane_wide_rows_total")
                wa, wd = val(m, "velogb10_lane_wide_accepted_total"), val(m, "velogb10_lane_wide_drafted_total")
                wsec = val(m, "velogb10_lane_wide_seconds_total")
                L.append("  ON on this boot" + (" (ALWAYS: every step with two or more eligible requests)" if wst > 1.5
                                                else " (the policy takes it only while its measured tokens/s beats today's arms)"))
                if wr > 0:
                    L.append(f"  wide rounds: {n(wr)} ({wl / wr:.1f} requests and {wrows / wr:.0f} rows per round); they produced {n(wt)} tokens"
                             + (f", {wt / wsec:.0f} tokens/s while running" if wsec > 0 else ""))
                    L.append(f"  drafts kept: {pct(wa, wd)} ({n(wa)} of {n(wd)})")
                    ph = [(p, val(m, "velogb10_lane_wide_phase_seconds_total", phase=p)) for p in ("draft", "trunk", "judge", "commit", "reprime")]
                    if wsec > 0:
                        L.append("  time per wide round: " + ", ".join(f"{p} {1e3 * v / wr:.1f} ms" for p, v in ph))
                else:
                    L.append("  NO wide round ran in this period")
                pk = {a: val(m, "velogb10_lane_wide_policy_steps_total", arm=a)
                      for a in ("wide", "other", "shared", "probe_wide", "probe_other", "probe_shared")}
                if sum(pk.values()) > 0:
                    L.append(f"  steps with two or more eligible requests: wide {n(pk['wide'])}, today's own arm choice {n(pk['other'])}, "
                             f"shared plain step {n(pk['shared'])}; measurement windows {n(pk['probe_wide'])} wide / "
                             f"{n(pk['probe_other'])} today's / {n(pk['probe_shared'])} shared")
                    L.append(f"  latest measured aggregate: wide {val(m, 'velogb10_lane_wide_measured_tok_s', arm='wide'):.0f} tok/s, "
                             f"today's arm {val(m, 'velogb10_lane_wide_measured_tok_s', arm='other'):.0f} tok/s, "
                             f"shared plain step {val(m, 'velogb10_lane_wide_measured_tok_s', arm='shared'):.0f} tok/s")
                why = {"penalty": "penalized request", "logprobs": "log probabilities asked", "image": "image request",
                       "sparse_regime": "past the sparse-attention switch", "control": "its periodic control / re-probe round",
                       "no_mtp": "speculation off for that request", "depth0": "no room for a draft", "prefill_interleave":
                       "a prompt was being read between steps", "too_few_lanes": "too few eligible requests (fewer than 3 with --lane-wide-verify on)", "rows_cap": "row cap",
                       "copy_stretch": "copy stretch"}
                held = [(r, val(m, "velogb10_lane_wide_fallback_total", reason=r)) for r in why]
                held = [(r, v) for r, v in held if v > 0]
                if held:
                    L.append("  request rounds kept out of wide rounds: " + "; ".join(f"{why[r]}: {n(v)}" for r, v in held))
            else:
                why = {3: "this is a TP server (TP=1 only in this version)", 4: "no MTP draft head",
                       5: "this pack's layer kinds are not wired for it", 6: "the DFlash2 chain is armed"}.get(int(round(wst)), "unknown reason")
                L.append(f"  asked for (--lane-wide-verify) but running OFF: {why}; today's rounds are unchanged")

        # LANEWIDE-DRES: --exl3-dense-resident: dense weights kept decoded for prefill-class passes.
        if ("velogb10_exl3_dense_resident_state", ()) in m:
            dst = val(m, "velogb10_exl3_dense_resident_state")
            L.append("")
            L.append("DENSE-RESIDENT (--exl3-dense-resident: dense weights decoded once and kept for prompt reading and wide verifies; "
                     "output unchanged)")
            if dst < 0.5:
                L.append("  off on this boot (the default)")
            elif val(m, "velogb10_exl3_dense_resident_matrices") == 0 and val(m, "velogb10_exl3_dense_resident_hits_total") == 0:
                L.append("  asked, but NOTHING is kept on this boot: the pack has no EXL3 dense matrix that prompt reading decodes "
                         "(FP4-Local serves its dense weights as stored), or the boot's warm-up was skipped; it changes nothing here")
            else:
                L.append(f"  ON: {n(val(m, 'velogb10_exl3_dense_resident_matrices'))} matrices kept, "
                         f"{val(m, 'velogb10_exl3_dense_resident_bytes') / 2**30:.2f} GiB; passes served from them: "
                         f"{n(val(m, 'velogb10_exl3_dense_resident_hits_total'))}")
                fb = [(r, val(m, "velogb10_exl3_dense_resident_fallback_total", reason=r)) for r in ("memory", "alloc", "capture")]
                fb = [(r, v) for r, v in fb if v > 0]
                if fb:
                    L.append("  decoded per pass instead (no room / allocation failed / graph capture): "
                             + "; ".join(f"{r}: {n(v)}" for r, v in fb))

        # POSTURE1: --exl3-sel-tc: the prefill sparse-attention gather on tensor cores (output may differ on near-ties)
        if ("velogb10_exl3_sel_tc_state", ()) in m:
            tst = val(m, "velogb10_exl3_sel_tc_state")
            L.append("")
            L.append("TENSOR-CORE PREFILL GATHER (--exl3-sel-tc: prompt reading past the sparse-attention switch runs its "
                     "gather on the tensor cores; OUTPUT MAY DIFFER ON NEAR-TIES from the default path)")
            if tst < 0.5:
                L.append("  off on this boot (the default)")
            else:
                fb = val(m, "velogb10_exl3_sel_tc_fallback") > 0.5
                if fb:
                    L.append(f"  ASKED but NOT served on this boot (the KV cache is not 8-bit, or this machine's head shape is not one the "
                             f"kernels take: 12 or 6 query heads per key head; the boot line says which): today's gather "
                             f"ran {n(val(m, 'velogb10_exl3_sel_tc_fallback_calls_total'))} time(s) instead")
                rows_tc = val(m, "velogb10_exl3_sel_tc_rows_total")
                if not fb: L.append(f"  ON: {n(val(m, 'velogb10_exl3_sel_tc_calls_total'))} gather launches, {n(rows_tc)} prompt rows "
                         "(summed over layers) served on the tensor cores"
                         + ("" if rows_tc > 0 else "; NONE yet (no prompt past the switch was read)"))
                c8 = val(m, "velogb10_exl3_sel_tc8_calls_total")
                if not fb and ("velogb10_exl3_sel_tc8_calls_total", ()) in m and c8 > 0:
                    L.append(f"  of those, {n(c8)} launches on the 6-query-heads-per-key-head kernel (TP=4: 8 prompt tokens per block)")

        # POSTURE1: --exl3-dense-pf-tc-qsa: the dense prompt rows of a sparse-attention pack on the tensor cores
        if ("velogb10_exl3_dense_pf_tc_qsa_state", ()) in m:
            L.append("")
            L.append("TENSOR-CORE DENSE PREFILL ON A SPARSE-ATTENTION PACK (--exl3-dense-pf-tc-qsa: the first 2,051 prompt tokens' "
                     "attention on the tensor cores; OUTPUT MAY DIFFER ON NEAR-TIES from the default path)")
            if val(m, "velogb10_exl3_dense_pf_tc_qsa_state") < 0.5:
                L.append("  off on this boot (the default)")
            else:
                q = val(m, "velogb10_exl3_dense_pf_tc_qsa_rows_total")
                L.append(f"  ON: {n(q)} dense prompt rows (summed over layers) on the tensor cores"
                         + ("" if q > 0 else "; NONE yet (no fresh prompt read since boot)"))

        # COPYREC1: --copy-span-record: a server-wide record of what earlier requests read and wrote; a lone request
        # whose own context has nothing to copy drafts from it (the model still checks every copied token).
        if ("velogb10_copy_span_record_state", ()) in m:
            rst = val(m, "velogb10_copy_span_record_state")
            # DELETE /v1/copy-record resets the record's counters with its ids: a period that spans an empty reads
            # the counters since the empty (never a negative difference)
            emp = val(m, "velogb10_copy_span_record_emptied_total") != 0
            for k in list(m):
                if k[0].startswith("velogb10_copy_span_record_") and k[0].endswith("_total") and m[k] < 0:
                    m[k] = 0.0
            L.append("")
            L.append("COPY RECORD (--copy-span-record: copies from what EARLIER requests read and wrote, when a request's own text "
                     "has nothing to copy; output is unchanged, only speed)")
            if emp:
                L.append("  (the record was emptied in this period, DELETE /v1/copy-record: its counters count from the empty)")
            if rst < 0.5:
                L.append("  not asked for on this boot (--copy-span-record off, the default)")
            elif rst < 1.5:
                held = val(m, "velogb10_copy_span_record_ids_held")
                mem = val(m, "velogb10_copy_span_record_memory_bytes") / 1048576
                disk = val(m, "velogb10_copy_span_record_disk_bytes") / 1048576
                cap = val(m, "velogb10_copy_span_record_cap_bytes") / 1048576
                where = ("written to its directory as it grows" if val(m, "velogb10_copy_span_record_persisting") > 0.5
                         else "kept in memory only (no directory, or persistence stopped: see the reasons below)")
                ro = " READ-ONLY: nothing is added;" if val(m, "velogb10_copy_span_record_read_only") > 0.5 else ""
                ondisk = f", {disk:.1f} MiB on disk" if disk > 0 else ""
                L.append(f"  ON on this boot;{ro} it holds {n(held)} token ids ({mem:.1f} MiB in memory{ondisk}, "
                         f"cap {cap:.0f} MiB), {where}")
                if held > 0:
                    L.append("  NOTE: a populated record makes repeated prompts faster than on a fresh server; benchmark with it "
                             "off or empty (DELETE /v1/copy-record)")
                lo = val(m, "velogb10_copy_span_record_ids_loaded")
                if lo > 0:
                    L.append(f"  loaded from disk at boot: {n(lo)} ids")
                ad, dd = val(m, "velogb10_copy_span_record_ids_added_total"), val(m, "velogb10_copy_span_record_ids_already_held_total")
                L.append(f"  ids added in this period: {n(ad)}; ids not added because the record held them already: {n(dd)}")
                rp, rpt = val(m, "velogb10_copy_span_record_proposals_total"), val(m, "velogb10_copy_span_record_proposed_tokens_total")
                ra, lk = val(m, "velogb10_copy_span_record_accepted_tokens_total"), val(m, "velogb10_copy_span_record_lookups_total")
                po, pr = val(m, "velogb10_copy_span_record_priced_out_total"), val(m, "velogb10_copy_span_record_probes_total")
                if rp > 0:
                    L.append(f"  copies started from the record: {n(rp)} (lookups {n(lk)}); copied tokens the model kept, over every "
                             f"round of those copies: {n(ra)} (first rounds offered {n(rpt)})")
                else:
                    L.append(f"  NO copy was started from the record in this period (lookups {n(lk)})")
                if po > 0 or pr > 0:
                    L.append(f"  asks held back because record copies were not paying: {n(po)}; probes let through anyway: {n(pr)} "
                             f"(one in {n(val(m, 'velogb10_copy_span_record_probe_every'))} now)")
                rates = [(b, val(m, "velogb10_copy_span_record_landing_rate", band=b)) for b in ("16_31", "32_63", "64_up")]
                L.append("  landing rate of a copied token by match length: "
                         + ", ".join(f"{b.replace('_up', '+').replace('_', '-')} tokens {r * 100:.0f}%" for b, r in rates))
                why = {"tokenizer_mismatch": "its directory belongs to another tokenizer (ran empty, wrote nothing there)",
                       "dir_error": "its directory could not be used (memory only)",
                       "locked": "another server writes its directory (read, not written)",
                       "load_error": "part of its files could not be read",
                       "disk_write": "a write failed (stopped persisting)",
                       "tp_digest": "the TP machines loaded different records (all started empty)",
                       "delete_error": "an empty could not delete a file"}
                fb = [(r, val(m, "velogb10_copy_span_record_fallback_total", reason=r)) for r in why]
                fb = [(r, v) for r, v in fb if v > 0]
                if fb:
                    L.append("  served with less than asked: " + "; ".join(f"{why[r]}: {n(v)}" for r, v in fb))
            else:
                L.append("  asked for (--copy-span-record on) but running OFF: --copy-span is not armed on this boot "
                         "(the record drafts through the copy arm); nothing is recorded")

        # DFLASH2-GEN: the nvfp4 trunk-path DF2 arm — engagement counters every serve path
        # exports (the exl3_df2_* series above are the EXL3 server's own arm).
        # TP2BIT5 #26: the two width-closure counters (Fix 2 + Fix 3). devroute_fallback moving
        # under TP = the boot asked for the host-routed arm and was served by the device-routed
        # EP arm instead (one clear boot line; never a refusal). ple_width_accepted = PLE rows
        # loaded outside the old {5,6} width set.
        drf = val(m, "velogb10_exl3_devroute_fallback_total")
        ple = val(m, "velogb10_exl3_ple_width_accepted_total")
        if drf > 0 or ple > 0:
            L.append("")
            L.append("EXL3 WIDTH CLOSURE (#26):")
            if drf > 0:
                L.append(f"  {n(drf)} prefill round(s) asked for the host-routed MoE arm under TP and were served "
                         f"by the device-routed EP arm instead (boot log: '[exl3] #26 fallback: ...')")
            if ple > 0:
                L.append(f"  {n(ple)} PLE n-gram row group(s) outside the old K in {{5,6}} set were loaded and served")
        # FP4L S5 (shared): the n-gram (PLE) table read-ahead — while a prefill chunk runs, the NEXT chunk's n-gram rows
        # are read into the page cache (only when the table stays on the SSD; a RAM-resident table needs none).
        wac = val(m, "velogb10_ple_warm_ahead_chunks_total")
        if wac > 0:
            L.append("")
            L.append("N-GRAM TABLE READ-AHEAD (prefill, table on the SSD):")
            L.append(f"  ENGAGED — {n(wac)} prefill chunk(s) had the next chunk's n-gram rows read ahead "
                     f"({n(val(m, 'velogb10_ple_warm_ahead_rows_total'))} rows, "
                     f"{n(val(m, 'velogb10_ple_warm_ahead_reads_total'))} page reads)")
        L += dsv41_engram_rows(m)
        L += dsv41_forward_rows(m)
        kvf = val(m, "velogb10_exl3_kv_cache_fallback_total")   # TPSHIP: --kv-cache bf16 on an EXL3 pack
        if kvf > 0:
            L.append("")
            L.append("EXL3 KV CACHE FORMAT:")
            L.append("  asked for --kv-cache bf16, which the EXL3 engine does not have: serving f16 instead (the same size per value)")
        wfb = val(m, "velogb10_exl3_pf_kernel_width_fallback_total")
        if wfb > 0:
            L.append("")
            L.append("EXL3 PREFILL KERNEL WIDTH DISPATCH (#26 Fix 3):")
            L.append(f"  {n(wfb)} prefill layer round(s) had a width the pipelined expert kernel does not decode (it does K 2..7 and 1.5/2.5/3.5) and were "
                     f"served by the wide tiles kernel instead (per-layer dispatch; boot log: '[exl3] #26 Fix 3: ...')")
        # K35 (2026-10-31): fractional expert widths (K = 1.5 / 2.5 / 3.5 = k2 3 / 5 / 7; turboderp's self-calibrated
        # 1.66 / 2.18 / 2.68 / 3.18 / 3.68 packs). The rounds counter includes graph replays; absent = never ran.
        frac_rounds = by_label(m, "velogb10_exl3_frac_expert_rounds_total", "k2")
        frac_rows = by_label(m, "velogb10_exl3_frac_pf_wide_rows_total", "k2")
        frac_pipe = by_label(m, "velogb10_exl3_frac_pf_pipelined_rows_total", "k2")   # PFHALF
        bridge = val(m, "velogb10_exl3_frac_pf_bridge_rows_total")
        if frac_rounds or frac_rows or frac_pipe or bridge > 0:
            L.append("")
            L.append("EXL3 FRACTIONAL EXPERT WIDTHS (K35: K = 1.5 / 2.5 / 3.5 routed experts):")
            for k2, v in sorted(frac_rounds.items()):
                L.append(f"  ENGAGED — K={int(k2) / 2:g} experts: {n(v)} MoE layer round(s) on the fractional grouped body "
                         f"(decode / verify / MTP draft; graph replays counted)")
            for k2, v in sorted(frac_pipe.items()):
                L.append(f"  ENGAGED — K={int(k2) / 2:g} experts: {n(v)} prefill expert row(s) (token x top-k) on the pipelined "
                         f"prefill kernel's fractional body (the fast path)")
            for k2, v in sorted(frac_rows.items()):
                L.append(f"  ENGAGED — K={int(k2) / 2:g} experts: {n(v)} prefill expert row(s) (token x top-k) on the fractional "
                         f"wide-tiles body (the slower fallback)")
            if bridge > 0:
                L.append(f"  {n(bridge)} prefill row(s) served by the LEG-B decode-moe bridge (the host-routed A/B arm only)")
        # K8EXP (2026-11-07): K = 7 / 8 experts (k2 14 / 16) — e.g. the 8-bit MTP head of the issue-#26 4bpw pack. The
        # rounds counter includes graph replays; absent = no K = 7 / 8 expert layer ran.
        k78 = by_label(m, "velogb10_exl3_k78_expert_rounds_total", "k2")
        if k78:
            L.append("")
            L.append("EXL3 K = 7 / 8 EXPERTS (K8EXP):")
            for k2, v in sorted(k78.items()):
                how = ("the plain grouped chain on its own K=8 kernel (no fused epilogue kernels at K=8)" if k2 == "16"
                       else "the integer expert family at K=7")
                L.append(f"  ENGAGED — K={int(k2) // 2} experts: {n(v)} MoE layer round(s) on {how} "
                         f"(decode / verify / MTP draft; graph replays counted)")

        # DSPLITK (owner 2026-11-07): the trunk's dense attention served split-K over the context (--exl3-dense-splitk on)
        pft = val(m, "velogb10_exl3_pf_tc_rows_total")
        if pft > 0:
            L.append("")
            L.append(f"EXL3 LONG-CONTEXT PREFILL (PFTC): ENGAGED — {n(pft)} prefill row(s) x attention layer(s) on the tensor-core kernels "
                     f"(--exl3-dense-pf-tc on)")
        dsk = val(m, "velogb10_exl3_dense_splitk_calls_total")
        if dsk > 0:
            L.append("")
            L.append(f"EXL3 LONG-CONTEXT ATTENTION (DSPLITK): ENGAGED — {n(dsk)} dense-attention launch(es) split over the context "
                     f"(--exl3-dense-splitk on; eager calls and graph captures, replays not counted)")

        df2v = val(m, "velogb10_df2_rounds_total")
        ovr = val(m, "velogb10_df2_mask_row_override")
        if df2v > 0 or ovr > 0:
            L.append("")
            L.append("DFLASH2 CHAIN (nvfp4 trunk-path arm):")
            if df2v > 0:
                L.append(f"  ENGAGED — {n(df2v)} DFlash2 draft round(s) served on this boot")
            if ovr >= 0.5:
                L.append("  mask row OVERRIDE engaged: the trunk's quantized mask row measured DAMAGED, "
                         "so MASK positions embed the artifact's exact BF16 copy")

        arms = by_label(m, "velogb10_lane_pack_rounds_total", "arm")
        total = sum(arms.values())
        # F1/F2/F3 (owner 2026-10-10): the boot's own decision about lane packing, from its gauge.
        lpa_rows = by_label(m, "velogb10_lane_pack_active", "reason")
        if lpa_rows:
            (reason, v), = lpa_rows.items()
            L.append("")
            if v >= 0.5:
                L.append("LANE PACKING IS ACTIVE on this boot (packed rounds can run)")
            elif reason == "--lane-pack off":
                L.append("LANE PACKING IS OFF on this boot (--lane-pack off was given; packing is on by default): every request runs its own round")
            else:
                L.append(f"LANE PACKING IS NOT ACTIVE on this boot (it is on by default): {reason}; requests are served without packed rounds")
        # FP4L (PLAN/FP4L_DESIGN.md §3a): an FP4-Local pack's precision levers as this boot resolved them, and the
        # engagement of its stored-precision kernels (host dispatches: in graphed decode these grow at capture).
        if val(m, "velogb10_fp4l_pack") >= 0.5:
            prec = next((dict(labels) for (name, labels) in m if name == "velogb10_fp4l_precision"), {})
            L.append("")
            L.append("FP4-LOCAL PACK (nvidia ModelOpt MXFP8 + NVFP4): precision levers on this boot")
            if prec:
                full = {"plain": "exact", "norms": "f32", "lm_head": "bf16", "draft_head": "head", "prefill_dense": "exact"}
                for k in ("plain", "norms", "lm_head", "draft_head", "prefill_dense"):
                    v = prec.get(k, "?")
                    tag = "full precision (stored values)" if full.get(k) == v else "SPEED end (rounds or re-quantizes stored values)"
                    if k == "draft_head":
                        tag = {"screen": "draft only, the default (int4 screen + exact rescore; the verify decides every token)",
                               "mxfp8": "draft only (the verify decides every token)",
                               "head": "draft only: the full draft-head rows (slower than the default screen)"}.get(v, tag)
                    L.append(f"  --fp4l-{k.replace('_', '-'):<14} {v:<6} {tag}")
                side = by_label(m, "velogb10_fp4l_dispatch_total", "weights").get("exl3_mtp_sidecar", 0)
                L.append("  MTP sidecar          " + ("ENGAGED: the draft passes' dense MTP linears come from an EXL3 sidecar (draft only)"
                                                     if side > 0 else "not in use (no mtp-*.safetensors in the model directory, or --fp4l-mtp-sidecar off)"))
            else:
                L.append("  (the precision gauge is not exposed: an older build)")
            tw = by_label(m, "velogb10_fp4l_twin_total", "op")
            if prec.get("plain") == "exact" or prec.get("norms") == "f32":
                if tw:
                    L.append("  stored-precision kernels dispatched (host launches; CUDA graph replays are not counted): "
                             + ", ".join(f"{op} {n(v)}" for op, v in sorted(tw.items())))
                elif window:
                    L.append("  no stored-precision kernel host launch in this window (graph replays are not counted; "
                             "run without a window for the since-boot count)")
                else:
                    L.append("  *** the full-precision levers are set but NO stored-precision kernel was ever dispatched ***")
        # F3: modules the wide trellis kernel cannot decode (widths outside 3..=5) served through the
        # reconstruct + cuBLASLt arm whatever --exl3-recon-min said (one boot line names it the first time).
        rf = by_label(m, "velogb10_exl3_recon_forced_total", "bits")
        if rf:
            det = ", ".join(f"{b}-bit: {n(v)}" for b, v in sorted(rf.items()))
            L.append(f"  this pack has modules the wide trellis kernel cannot decode; those GEMMs serve through the "
                     f"reconstruct arm (--exl3-recon-min overridden): {det}")
        L.append("")
        L.append("HOW ROUNDS WERE SERVED (every round counted in this period, ALSO the ones run while only one request was busy)")
        if total > 0:
            packed, serial, shared = arms.get("packed", 0), arms.get("serial", 0), arms.get("shared_plain", 0)
            L.append(f"  packed       {n(packed):>9}  {pct(packed, total):>4}   ONE round that served TWO requests (the lane feature); it counts once here")
            L.append(f"  serial       {n(serial):>9}  {pct(serial, total):>4}   one request's own speculative round: the others wait their turn, or it was running alone")
            L.append(f"  shared plain {n(shared):>9}  {pct(shared, total):>4}   all busy requests in one step, speculation OFF for that step")
        else:
            L.append("  (no round counted by arm: an older build, or nothing ran)")

        skips = {k: v for k, v in by_label(m, "velogb10_lane_pack_skip_total", "reason").items() if v > 0}
        L.append("")
        packed = arms.get("packed", 0)
        if packed > 0:
            L.append(f"LANE PACKING RAN: {n(packed)} packed rounds = {n(2 * packed)} request-rounds served two at a time, "
                     f"against {n(arms.get('serial', 0))} served one at a time (that figure includes every round of a request that ran alone, "
                     f"so this is NOT the share of the time two requests overlapped: the server does not count that).")
        else:
            L.append("LANE PACKING DID NOT RUN: 0 packed rounds." + ("" if skips or total else
                     " (Were two requests ever busy at once? Packing needs two busy requests; a single request never packs.)"))
        if val(m, "velogb10_lane_pack_disabled") > 0:
            L.append("  *** the packed arm switched itself OFF for the rest of this boot after an error in a packed round ***")
        if skips:
            L.append("  times a request was kept OUT of a packed round, by reason:")
            for k, v in sorted(skips.items(), key=lambda kv: -kv[1]):
                L.append(f"    {k:<13}{n(v):>8}   {SKIP_WORDS.get(k, '(no description: see src/lane_pack.rs)')}")
    # F7b (2026-10-11): unknown "--"-options ignored at boot (warn, never refuse)
    # S-C4XOVER: the spec-lanes auto policy's own decisions (estimates, scale, flips)
    # POSTURE1: the key's labels are a tuple of (name, value) PAIRS (parse); the old test compared against one bare pair, so
    # this section never printed
    if ("velogb10_spec_lanes_est_tok_s", (("arm", "serial"),)) in m or ("velogb10_spec_lanes_est_tok_s", (("arm", "shared"),)) in m:
        es = val(m, "velogb10_spec_lanes_est_tok_s", arm="serial")
        eh = val(m, "velogb10_spec_lanes_est_tok_s", arm="shared")
        dec = val(m, "velogb10_spec_lanes_decisions_total")
        sc = val(m, "velogb10_spec_lanes_shared_scale")
        busy = val(m, "velogb10_spec_lanes_busy_lanes")
        sw_s = val(m, "velogb10_spec_lanes_switches_total", to="shared")
        sw_r = val(m, "velogb10_spec_lanes_switches_total", to="serial")
        L.append("")
        L.append("SPEC-LANES AUTO POLICY (the arm-choice between shared plain steps and requests taking turns on speculation)")
        if dec == 0:
            L.append("  no decision compared real measurements yet this boot (the policy rode its fallback threshold, "
                     "or only one request was busy)")
        else:
            L.append(f"  at its last measured decision: {n(busy)} busy lanes, est serial {es:.0f} tok/s vs "
                     f"est shared {eh:.0f} tok/s (shared step cost model corrected by learned scale {sc:.2f})")
            if es > 0 and eh > 0:
                better = "serial speculation" if es > eh else "shared plain steps"
                L.append(f"  the estimate favored: {better}")
        L.append(f"  pick flips since boot: to-shared {n(sw_s)}, to-serial {n(sw_r)}"
                 + (" (a large count says the policy is oscillating between arms)" if sw_s + sw_r > 10 else ""))
        ms = val(m, "velogb10_spec_lanes_measured_tok_s", arm="serial")
        mh = val(m, "velogb10_spec_lanes_measured_tok_s", arm="shared")
        L.append(f"    measured delivery: serial {ms:.0f} tok/s, shared {mh:.0f} tok/s "
                 f"(what the arms actually delivered; the pick compares these when both fresh)")
        hms, sms = val(m, "velogb10_spec_lanes_arm_ms_total", arm="shared"), val(m, "velogb10_spec_lanes_arm_ms_total", arm="serial")
        htk, stk = val(m, "velogb10_spec_lanes_arm_tok_total", arm="shared"), val(m, "velogb10_spec_lanes_arm_tok_total", arm="serial")
        if hms + sms > 0:
            L.append(f"    metered time: shared plain {pct(hms, hms + sms)} ({htk / max(hms, 1e-9) * 1e3:.0f} tok/s), speculation "
                     f"{pct(sms, hms + sms)} ({stk / max(sms, 1e-9) * 1e3:.0f} tok/s)")
        mv = val(m, "velogb10_spec_lanes_meter")
        if ("velogb10_spec_lanes_meter", ()) in m:
            L.append("    how the arms are measured: " + ("v2 (--spec-arm-meter v2): whole steps, at the exact number of busy requests, "
                     "never while a prompt is being read" if mv == 2 else "v1 (the default): each arm's own rounds, rates shared "
                     "between busy counts that differ by one; --spec-arm-meter v2 measures whole steps per exact count"))
        tps = val(m, "velogb10_spec_lanes_meter_tp_sync_total")
        if mv == 2 and tps > 0:
            L.append(f"    under TP: {n(tps)} metered steps took the head's step time on every machine (the machines pick the same arm)")
        rpl = val(m, "velogb10_spec_lanes_meter_replace_total")
        if mv == 2:
            L.append(f"    probe measurements that replaced a contradicted rate (off by more than 15%): {n(rpl)}")
        pb = val(m, "velogb10_spec_lanes_probes_total", arm="shared")
        ps = val(m, "velogb10_spec_lanes_probes_total", arm="serial")
        L.append(f"    arm-rate probes: shared {n(pb)}, serial {n(ps)} (short windows of the idle arm, "
                 f"to keep both measurements fresh)")
    # F7b (2026-10-11): unknown "--"-options ignored at boot (warn, never refuse)
    uo = val(m, "velogb10_unknown_options")
    if ("velogb10_unknown_options", ()) not in m:
        # found 2026-10-12 by the v0.7.4 cut: the builds up to and including the release candidate set the count but
        # never render it, so an absent series must not read as "all clear". The startup warning line is the signal.
        L.append("  unknown-option count: NOT EXPORTED by this build (the 'unknown option' warning lines at startup are the only signal; "
                 "absence of the series is not an all-clear)")
    elif uo == 0:
        L.append("  every flag on the command line was known (0 unknown options ignored)")
    else:
        L.append(f"  unknown options ignored: {n(uo)} - each got one '[opts] unknown option' line at boot (with up to three near names); a misspelt flag used to disappear silently")
    # the tool-call parser's shadow counting (S-B9-TOOLPARSE3 T1): the counters say what the OTHER rules
    # of the parser would have returned for the text it parsed; nothing here changes what is served
    blocks = val(m, "velogb10_toolcall_blocks_total")
    served = next((dict(labels) for (name, labels) in m if name == "velogb10_toolcall_rules_info"), None)
    if served is not None or blocks > 0:
        L.append("")
        L.append("TOOL-CALL PARSER (how the text of tool calls was read; the shadow counters only compare, they change nothing)")
        if served is not None:
            L.append(f"  served rules: values end by --tool-quote-pairing {served.get('pairing', '?')}, "
                     f"blocks end by --tool-block-end {served.get('block', '?')}, "
                     f"unclosed blocks by --tool-unclosed {served.get('unclosed', '?')}"
                     + (" (drop = the v0.7.3 reading, the default: a block the model never closed is dropped whole; "
                        "--tool-unclosed complete returns its closed function elements when the model stopped by itself)"
                        if served.get('unclosed') == 'drop' else ""))
        L.append(f"  <tool_call> blocks read: {n(blocks)}")
        unc = by_label(m, "velogb10_toolcall_unclosed_total", "outcome")
        if unc:
            parts = ", ".join(f"{k}={n(v)}" for k, v in sorted(unc.items()))
            L.append(f"  blocks never closed: {parts} (returned = completed on a self-stop; "
                     f"dropped_cut = self-stop with nothing complete; dropped_length = cut by budget/client/engine)")
        skipped = by_label(m, "velogb10_toolcall_shadow_skipped_total", "reason")
        if sum(skipped.values()) > 0:
            L.append("  the shadow comparison was NOT made or was cut short for some responses: " + ", ".join(
                f"{k} {n(v)}" for k, v in sorted(skipped.items()) if v) + " (size = text over 512 KiB, blocks = more than 64 blocks, "
                "time = the comparison's 100 ms wall-clock budget, spent between two rules); what was served is unaffected")
        declined = by_label(m, "velogb10_toolcall_cost_declined_total", "role")
        if declined.get("served", 0) > 0:
            L.append(f"  the cost model was SERVED and read {n(declined['served'])} block(s) by the v0.7.3 rule because they exceeded its work "
                     f"budget (60,000 work units per block, 200,000 per response, about 4 ms of one core): those blocks got the v0.7.3 reading")
        if declined.get("shadow", 0) > 0:
            L.append(f"  the cost model, read only to compare, skipped {n(declined['shadow'])} block(s) over its work budget "
                     f"(not compared; what was served is unaffected)")
        # how long the thread that parses a response was busy with it (on the EXL3 server that thread serves
        # every stream, /health and /metrics: this is how long they waited)
        edges = sorted((float(dict(l)["le"]), v) for (nm, l), v in m.items()
                       if nm == "velogb10_toolcall_parse_seconds_bucket" and dict(l).get("part") == "served" and dict(l).get("le") != "+Inf")
        for part in ("served", "shadow"):
            cnt = val(m, "velogb10_toolcall_parse_seconds_count", part=part)
            if cnt <= 0:
                continue
            bk = sorted((float(dict(l)["le"]), v) for (nm, l), v in m.items()
                        if nm == "velogb10_toolcall_parse_seconds_bucket" and dict(l).get("part") == part and dict(l).get("le") != "+Inf")
            def quant(q):
                for le, c in bk:
                    if c >= q * cnt:
                        return f"<= {le * 1e3:g} ms"
                return "> " + (f"{bk[-1][0] * 1e3:g} ms" if bk else "?")
            mean = val(m, "velogb10_toolcall_parse_seconds_sum", part=part) / cnt
            worst = val(m, "velogb10_toolcall_parse_seconds_max", part=part)
            L.append(f"  time in the parser, {part}: {n(cnt)} responses, mean {mean * 1e3:.3f} ms, median {quant(0.5)}, 99th {quant(0.99)}, "
                     f"worst {worst * 1e3:.3f} ms" + ("  (the thread that serves every stream waited this long)" if part == "served" else
                     "  (the comparison that is only counted)"))
        unclosed = {}
        for (name, labels), v in m.items():
            if name == "velogb10_toolcall_unclosed_total" and v > 0:
                d = dict(labels)
                unclosed[(d.get("finish"), d.get("ends_like_call"))] = v
        if unclosed:
            stop_call = unclosed.get(("stop", "yes"), 0)
            L.append(f"  responses that ENDED INSIDE a <tool_call> block the model never closed (dropped under every rule): "
                     f"{n(sum(unclosed.values()))}; of those {n(stop_call)} stopped by themselves with text that ends like a finished call "
                     f"(a forgotten close token, or a call drafted inside reasoning)")
        if val(m, "velogb10_toolcall_shadow_failed_total") > 0:
            L.append(f"  *** the shadow comparison panicked {n(val(m, 'velogb10_toolcall_shadow_failed_total'))} time(s) (a defect in a rule that is not served; "
                     f"what was served is unaffected) ***")
        if blocks == 0:
            L.append("  no block was read in this period (no request that offered tools produced a call).")
        else:
            words = {"off": "the v0.7.3 value rule (--tool-quote-pairing off)", "on": "quote pairing (--tool-quote-pairing on)",
                     "cost": "the cost model (--tool-quote-pairing cost)", "block_end": "the other block-end rule (--tool-block-end)"}
            comp = by_label(m, "velogb10_toolcall_shadow_compared_total", "other")
            dis = {}
            for (name, labels), v in m.items():
                if name == "velogb10_toolcall_parse_disagreement_total" and v > 0:
                    d = dict(labels)
                    dis[(d.get("other"), d.get("kind"), d.get("fewer"))] = v
            if comp.get("cost", 0) <= 0 and (served or {}).get("pairing") != "cost":
                L.append("  the cost model was NOT read in the shadow (--tool-shadow-cost off, the default: reading it costs the thread that "
                         "parses the response up to ~4 ms per response); the other rules below were")
            for other in ("off", "on", "cost", "block_end"):
                if comp.get(other, 0) <= 0:
                    continue
                tot = sum(v for (o, _, _), v in dis.items() if o == other)
                L.append(f"  also read under {words[other]}: {n(comp[other])}; it read the text differently in {n(tot)} ({pct(tot, comp[other])})")
                for (o, kind, fewer), v in sorted(dis.items()):
                    if o != other:
                        continue
                    what = {"call_count": "a different NUMBER of calls", "arg_boundary": "the same number of calls, an argument or name differs",
                            "content": "the same calls, the prose differs"}.get(kind, kind)
                    who = {"served": "the SERVED rule returned fewer calls", "other": "the OTHER rule returned fewer calls",
                           "same": "equal number of calls"}.get(fewer, fewer)
                    L.append(f"      {n(v):>7}  {what}; {who}")
            if dis:
                L.append("  A difference means the text was ambiguous under the rules (a value that quotes tool-call syntax). It does not say "
                         "which reading is right; the server log has one content-free line per difference ([toolcall-shadow], block hash and sizes only).")
            else:
                L.append("  every other rule read every block the same way: nothing in this traffic quoted tool-call syntax.")
    for name, words in (("velogb10_graph_capture_invalidated_total",
                         "CUDA graph captures lost to a CUDA call that is illegal during capture (another thread's, or the capturing thread's own; the step ran eagerly)"),
                        ("velogb10_graph_capture_failed_total",
                         "CUDA graph captures that produced no graph (BeginCapture, EndCapture, instantiate or the TP graph audit; the step ran eagerly)"),
                        ("velogb10_requests_rejected_total", "requests refused at admission"),
                        ("velogb10_streams_cancelled_backlog_total", "streams cancelled because the reader stalled")):
        if val(m, name) > 0:
            L.append(f"NOTE: {words}: {n(val(m, name))}")
    # S-C4PRIMEA (2026-10-15): the batched shared-step head-row prime path.
    pb = val(m, "velogb10_mtp_prime_batch_total")
    if pb is not None and not naive:
        pbl = val(m, "velogb10_mtp_prime_batch_lanes_total") or 0
        if pb > 0:
            L.append("")
            L.append(f"PRIME BATCHING RAN: {n(pb)} shared-step primes served by the ONE-LAUNCH path "
                     f"({n(pbl)} head-row lanes; --exl3-prime-batch off disables it)")
        else:
            L.append("")
            L.append("PRIME BATCHING DID NOT RUN: 0 batched primes (no multi-lane shared step ran, or --exl3-prime-batch off)")
    # Naive-N0.5 (2026-10-20): how its TP decode reduced, how its experts are sharded, how many requests share a step,
    # and how its speculation did on sampled requests
    if any(k[0].startswith("velogb10_naive_") for k in m):
        L.append("")
        L.append("NAIVE-N0.5:")
        if ("velogb10_naive_tp_reduce_single_stage", ()) in m:
            single = val(m, "velogb10_naive_tp_reduce_single_stage") > 0
            L.append(f"  decode all-reduce: {'ONE all-peers stage (TP=4 single-stage ctx)' if single else 'the recursive-doubling tree'}"
                     + (f"; {n(val(m, 'velogb10_naive_tp_reduce_single_epochs_total'))} single-stage reduces" if single else "")
                     + f"; {n(val(m, 'velogb10_naive_tp_reduce_tree_epochs_total'))} tree exchanges (prefill chunks{'' if single else ' and every decode reduce'})")
            if single:
                L.append(f"  K1 copy: {'multi-block' if val(m, 'velogb10_naive_tp_reduce_single_k1_multiblock') > 0 else 'single-block'}; receive "
                         f"{'GPU-validated (--tp-gpu-recv on)' if val(m, 'velogb10_naive_tp_reduce_single_gpu_recv') > 0 else 'CPU bounce (default)'}")
        if ("velogb10_naive_lanes", ()) in m:
            tp = val(m, "velogb10_naive_moe_shard_tp") > 0
            L.append(f"  experts: {'TENSOR-parallel (every rank holds a slice of every expert; --naive-moe-shard, the TP=4 default)' if tp else 'expert-parallel (whole experts per rank; --naive-moe-shard, the TP=2 default)'}")
            ln = int(val(m, "velogb10_naive_lanes"))
            L.append(f"  concurrent requests: {ln} ({'one at a time' if ln <= 1 else 'up to ' + str(ln) + ' share each step, --max-batch'})")
            if ln > 1 and ("velogb10_naive_lane_ctx", ()) in m:
                L.append("  drafter context per lane: " + ("ON (any request left alone speculates; --naive-lane-ctx)" if val(m, "velogb10_naive_lane_ctx") > 0
                         else "OFF (only a request admitted into an idle engine can speculate)"))
        if ("velogb10_naive_prefix_state", ()) in m:
            ps = int(val(m, "velogb10_naive_prefix_state"))
            h, mi, rt = val(m, "velogb10_naive_prefix_hits_total"), val(m, "velogb10_naive_prefix_misses_total"), val(m, "velogb10_naive_prefix_reused_tokens_total")
            if ps == 1:
                L.append(f"  prefix reuse: ON; {n(h)} prompts resumed after the previous prompt's prefix ({n(rt)} prompt tokens not read again), "
                         f"{n(mi)} prompts read whole")
            elif ps == 2:
                L.append("  prefix reuse: OFF on this boot although --prefix-cache on was asked (an older build: single-slot, --naive-lanes 1 only); every prompt is read whole")
            else:
                L.append("  prefix reuse: off (--prefix-cache not on); every prompt is read whole")
        r, a = val(m, "velogb10_naive_spec_sampled_rounds_total"), val(m, "velogb10_naive_spec_sampled_accepted_total")
        if r > 0:
            L.append(f"  speculation on sampled requests: {n(r)} rounds, {n(a)} drafts accepted ({a / r:.2f} per round)")
        else:
            L.append("  speculation on sampled requests: no sampled round ran (speculation off on this boot, e.g. --spec-source none, or no sampled request yet)")
    return L


def dsv41_spec_rows(m):
    """DeepSeek-V4.1 DSpark speculation in plain words from its own counters (src/dsv41/gpu/dspark/metrics.rs).
    Empty when no DSpark round ran and no source was announced in the server."""
    L = []
    # DSV41 DS: DSpark speculation (velogb10_dsv41_spec_*). Absent = no DSpark round ran and no source was announced.
    if any(k[0] == "velogb10_dsv41_spec_rounds_total" for k in m):
        src = [dict(k[1]).get("source", "?") for k in m if k[0] == "velogb10_dsv41_spec_source"]
        rd, dr, ac = val(m, "velogb10_dsv41_spec_rounds_total"), val(m, "velogb10_dsv41_spec_drafted_total"), val(m, "velogb10_dsv41_spec_accepted_total")
        L.append("")
        L.append("DEEPSEEK-V4.1 SPECULATION (DSpark):")
        L.append(f"  source: {src[0] if src else '?'}")
        pol = [dict(k[1]).get("policy", "?") for k in m if k[0] == "velogb10_dsv41_spec_policy"]
        k0 = val(m, "velogb10_dsv41_spec_draft_len_total", k="0")
        if pol and pol[0] == "off":
            # INT8: --mtp off / --dsv41-draft-len 0. The serve loop still counts its plain decode steps as rounds of 0 drafts;
            # calling that ENGAGED misled (UXF boot B: "ENGAGED: 729 round(s) ... 0 of 0 drafts accepted").
            L.append(f"  SPECULATION IS OFF on this server (draft length 0: --mtp off or --dsv41-draft-len 0): plain decoding, one token per step; "
                     f"{n(rd)} decode step(s), {n(dr)} drafts made or verified")
            if dr > 0:
                L.append(f"  !! {n(dr)} draft(s) were verified although the policy is off: the counters disagree, report it")
            L += dsv41_markov_rows(m)
        elif rd > 0 and dr == 0:
            why = (f"the {pol[0]} draft-length policy chose 0 drafts on every round" if pol else
                   "draft length 0 on every round: --mtp off or --dsv41-draft-len 0, or the auto policy chose 0; this server does not publish its policy")
            L.append(f"  SPECULATION ENGAGED 0 TIMES: {n(rd)} round(s), none verified a draft ({why}): plain decoding, one token per step")
        elif rd > 0:
            L.append(f"  ENGAGED: {n(rd)} round(s) ({n(val(m, 'velogb10_dsv41_spec_sampled_rounds_total'))} sampled, {n(val(m, 'velogb10_dsv41_spec_greedy_rounds_total'))} greedy); "
                     f"{n(ac)} of {n(dr)} drafts accepted ({pct(ac, dr)}); {1 + ac / rd:.2f} tokens per round")
            pos, off = {}, {}
            for (nm, lb), x in m.items():
                if nm == "velogb10_dsv41_spec_position_total":
                    d = dict(lb)
                    tgt = pos if d.get("outcome") == "accepted" else off
                    tgt[d.get("position", "?")] = tgt.get(d.get("position", "?"), 0.0) + x
            if off:
                L.append("  acceptance by position: " + ", ".join(f"#{p}: {pct(pos.get(p, 0), off[p])}" for p in sorted(off, key=int) if off[p] > 0))
            L.append(f"  after a rejection: {n(val(m, 'velogb10_dsv41_spec_residual_draws_total'))} residual draw(s), {n(val(m, 'velogb10_dsv41_spec_zero_mass_total'))} empty residual(s) (first max logit taken)")
            L += dsv41_spec_rule_rows(m)
            L += dsv41_markov_rows(m)
            if k0 > 0:
                L.append(f"  {n(k0)} of the {n(rd)} round(s) verified no draft (the draft-length policy chose 0 for them: plain steps)")
            wc = val(m, "velogb10_dsv41_spec_wide_cut_rows_total")
            if wc > 0:
                L.append(f"  {n(wc)} row(s) needed their top-k / top-p cut computed off the device (slower, counted)")
        else:
            L.append("  SPECULATION ENGAGED 0 TIMES: the source is announced but no DSpark round ran")
    return L


def dsv41_markov_rows(m):
    """Leg PH1: the drafter's Markov bias cache (src/dsv41/gpu/dspark/mcache.rs): how many draft steps skipped the Markov
    head read because every active request's chain input token had a kept row. The drafts are the same either way."""
    L = []
    pf = val(m, "velogb10_dsv41_spec_plain_finish_total")
    if pf > 0:
        L.append(f"  {n(pf)} verify(s) of 0 drafts finished without the residual pass (one exchange fewer each; no request can reject there)")
    off = [dict(k[1]).get("reason", "?") for k in m if k[0] == "velogb10_dsv41_markov_cache_off"]
    if off:
        return L + [f"  drafter Markov cache: OFF ({off[0]}); every draft step reads the Markov head"]
    st = [dict(k[1]) for k in m if k[0] == "velogb10_dsv41_markov_cache_tokens"]
    if not st:
        return L
    d = st[0]
    k = val(m, "velogb10_dsv41_markov_cache_tokens", **d)
    dyn_cap = val(m, "velogb10_dsv41_markov_dyn_cap")
    dyn_f = val(m, "velogb10_dsv41_markov_dyn_fills_total")
    eskip = val(m, "velogb10_dsv41_markov_cache_early_skip_steps") or val(m, "velogb10_dsv41_markov_device_early_skipped_total")
    h, c = val(m, "velogb10_dsv41_markov_steps_cached_total"), val(m, "velogb10_dsv41_markov_steps_computed_total")
    ih, iu = val(m, "velogb10_dsv41_markov_inputs_cached_total"), val(m, "velogb10_dsv41_markov_inputs_uncached_total")
    dh, dc = val(m, "velogb10_dsv41_markov_device_steps_cached_total"), val(m, "velogb10_dsv41_markov_device_steps_computed_total")
    chk = f"load check {d.get('check_groups', '?')} token groups, {d.get('check_mismatch', '?')} mismatching"
    dyn_desc = f" + {n(dyn_cap)} dynamic ring slots" if dyn_cap > 0 else ""
    if h + c == 0 and eskip == 0:
        L.append(f"  drafter Markov cache: {n(k)} static{dyn_desc} tokens ({d.get('list', '?')}; {d.get('mb', '?')} MB a rank; {chk}); ENGAGED 0 TIMES: no draft step ran yet")
        return L + dsv41_copy_rows(m)
    L.append(f"  drafter Markov cache: ENGAGED on {n(h)} of {n(h + c)} draft steps ({pct(h, h + c)}: the Markov head read skipped); "
             f"{pct(ih, ih + iu)} of chain input tokens cached; {n(k)} static{dyn_desc} tokens ({d.get('list', '?')}; {d.get('mb', '?')} MB a rank; {chk})")
    if dyn_cap > 0 or dyn_f > 0:
        L.append(f"  dynamic Markov on-miss GPU writebacks (--dsv41-markov-dyn-cache): {n(dyn_f)} session-vocabulary row(s) cached across {n(dyn_cap)} ring slots")
    if ("velogb10_dsv41_markov_cache_early_skip_steps", ()) in m or eskip > 0:
        L.append(f"  per-step confidence early-stop (--dsv41-draft-early-stop): {n(eskip)} Markov head step(s) skipped after exact 1-lane stopping rule halted")
    by = {}
    for (nm, lb), x in m.items():
        if nm == "velogb10_dsv41_markov_steps_by_lanes_total":
            e = dict(lb)
            by.setdefault(e.get("lanes", "?"), [0.0, 0.0])[0 if e.get("outcome") == "cached" else 1] += x
    rows = [f"{ln} request(s) {pct(v[0], v[0] + v[1])} of {n(v[0] + v[1])}" for ln, v in sorted(by.items()) if v[0] + v[1] > 0]
    if rows:
        L.append("  cached steps by concurrent requests: " + ", ".join(rows))
    if dh + dc > 0:
        L.append(f"  the kernel's own count (read every 64 rounds): {n(dh)} cached, {n(dc)} computed, "
                 f"{n(val(m, 'velogb10_dsv41_markov_device_dyn_fills_total'))} dyn fills, {n(val(m, 'velogb10_dsv41_markov_device_early_skipped_total'))} early-skipped")
    if d.get("check_mismatch", "0") not in ("0", "?"):
        L.append("  !! the load-time check found cached rows that differ from the computed step: report it")
    return L + dsv41_copy_rows(m)


def dsv41_copy_rows(m):
    """Leg CD1: exact suffix copy drafting (--dsv41-copy-draft on) inside DSpark speculation."""
    L = []
    if ("velogb10_dsv41_copy_draft_min_match", ()) not in m:
        return L
    mm = val(m, "velogb10_dsv41_copy_draft_min_match")
    cr, lr = val(m, "velogb10_dsv41_copy_rounds_total"), val(m, "velogb10_dsv41_copy_lane_rounds_total")
    pr, ac = val(m, "velogb10_dsv41_copy_proposed_total"), val(m, "velogb10_dsv41_copy_accepted_total")
    sk, si = val(m, "velogb10_dsv41_copy_draft_pass_skipped_total"), val(m, "velogb10_dsv41_copy_silenced_total")
    L.append(f"  copy drafting (--dsv41-copy-draft on, min match {n(mm)} tokens): {n(cr)} copy round(s) ({n(lr)} lane-round(s)), "
             f"{n(ac)} of {n(pr)} copied tokens accepted ({pct(ac, pr)}), {n(sk)} DSpark draft pass(es) skipped entirely, {n(si)} yield-gate silence(s)")
    return L


def dsv41_spec_rule_rows(m):
    """Leg PS1: how the auto draft-length rule chose (src/dsv41/gpu/dspark/metrics.rs render_ps1): the rule, the drafts it
    chose per round, the tokens it expected against those it got, the round time, and the cost table it priced with."""
    L = []
    rule = [dict(k[1]).get("rule", "?") for k in m if k[0] == "velogb10_dsv41_spec_rule"]
    tw, tn = val(m, "velogb10_dsv41_spec_round_wall_ms_total"), val(m, "velogb10_dsv41_spec_timed_rounds_total")
    if tn > 0:
        L.append(f"  round time (draft + choice + verify): {tw / tn:.1f} ms on average over {n(tn)} drafted round(s)")
    if rule:
        what = {"conf": "conf = the baseline recipe's per-round rule (drafter confidences + measured round costs)",
                "stats": "stats = acceptance by position + measured round costs"}.get(rule[0], rule[0])
        L.append(f"  auto draft-length rule: {what}")
    ch = {dict(k[1]).get("k", "?"): x for k, x in m.items() if k[0] == "velogb10_dsv41_spec_chosen_k_total"}
    dec = val(m, "velogb10_dsv41_spec_decided_rounds_total")
    if dec > 0:
        L.append("  drafts chosen per round: " + ", ".join(f"{k}: {n(ch[k])}" for k in sorted(ch, key=int) if ch[k] > 0)
                 + f" (mean {sum(int(k) * x for k, x in ch.items()) / dec:.2f})")
        ex, re = val(m, "velogb10_dsv41_spec_expected_tokens_total"), val(m, "velogb10_dsv41_spec_realised_tokens_total")
        L.append(f"  tokens per decided round: expected {ex / dec:.2f}, got {re / dec:.2f} (summed over the round's requests; {n(dec)} round(s))")
    pr = val(m, "velogb10_dsv41_spec_probes_total")
    if pr > 0:
        L.append(f"  {n(pr)} probe round(s) offered all 5 drafts to re-measure deep positions (stats rule)")
    tab = {}
    for (nm, lb), x in m.items():
        if nm == "velogb10_dsv41_spec_round_cost_ms":
            d = dict(lb)
            tab.setdefault(d.get("lanes", "?"), {})[d.get("k", "?")] = x
    ov = {dict(k[1]).get("lanes", "?"): x for k, x in m.items() if k[0] == "velogb10_dsv41_spec_overhead_ms"}
    for ln in sorted(tab, key=int):
        row = " ".join(f"{k}:{tab[ln][k]:.1f}" for k in sorted(tab[ln], key=int))
        L.append(f"  priced round ms at {ln} request(s), by drafts per request: {row}" + (f"; learned overhead {ov[ln]:.1f} ms" if ln in ov else ""))
    L += dsv41_cost_model_rows(m)
    L += dsv41_alloc_rows(m)
    return L


def dsv41_alloc_rows(m):
    """Leg PS3: how a round's verify rows were shared between its requests (src/dsv41/gpu/dspark/alloc.rs render): the mode,
    the row cap, per number of requests in a round the rounds, the verify rows and the tokens expected against those got,
    and the drafts each request verified."""
    L = []
    mode = [dict(k[1]) for k in m if k[0] == "velogb10_dsv41_spec_alloc"]
    if mode:
        d = mode[0]
        what = {"lanes": "each request its own count by landing chance (the baseline recipe's lane allocation)",
                "shared": "one draft count for every request of a round"}.get(d.get("mode", "?"), d.get("mode", "?"))
        cap = d.get("max_rows", "?")
        asked = d.get("max_rows_asked", cap)
        L.append(f"  rows of a round with several requests: {what}; at most {cap} verify rows a round"
                 + (f" (asked {asked}: this engine's round carries at most 32)" if asked != cap else ""))
    st = [dict(k[1]).get("rule", "?") for k in m if k[0] == "velogb10_dsv41_spec_stop_rule"]
    if st:
        if val(m, "velogb10_dsv41_spec_inexact_scan") > 0:
            L.append("  !! DRAFT COUNT BY THE INEXACT SCAN (--dsv41-spec-inexact-scan on): sampled output is NOT exactly the model's distribution; a measurement build, not for use")
        else:
            L.append(f"  draft count cut by an exact rule ({st[0]}): whether a draft is verified never depends on that draft's own draw (sampled output exact)")
    per = {}
    for (nm, lb), x in m.items():
        if nm.startswith("velogb10_dsv41_spec_lanes_") and nm.endswith("_total"):
            per.setdefault(dict(lb).get("lanes", "?"), {})[nm[len("velogb10_dsv41_spec_lanes_"):-len("_total")]] = x
    hist = {}
    for (nm, lb), x in m.items():
        if nm == "velogb10_dsv41_spec_lane_drafts_total":
            d = dict(lb)
            hist.setdefault(d.get("lanes", "?"), {})[d.get("c", "?")] = x
    for ln in sorted(per, key=int):
        e = per[ln]
        r = e.get("rounds", 0)
        if r <= 0:
            continue
        h = hist.get(ln, {})
        L.append(f"  {ln} request(s) in a round: {n(r)} round(s), {e.get('rows', 0) / r:.1f} verify rows a round, tokens a round expected "
                 f"{e.get('expected_tokens', 0) / r:.2f}, got {e.get('realised_tokens', 0) / r:.2f}; drafts verified per request: "
                 + ", ".join(f"{c}: {n(h[c])}" for c in sorted(h, key=int) if h[c] > 0))
    un, la, sc, up = (val(m, "velogb10_dsv41_spec_uneven_rounds_total"), val(m, "velogb10_dsv41_spec_alloc_lookahead_total"),
                      val(m, "velogb10_dsv41_spec_slack_capped_total"), val(m, "velogb10_dsv41_spec_alloc_unpriced_total"))
    if mode or un > 0:
        L.append(f"  {n(un)} round(s) gave their requests different counts; {n(sc)} request-round(s) offered at most their last count + 2; "
                 f"{n(la)} round(s) where the inexact scan (the best count over all counts) would have chosen another count than the exact rule served"
                 + (f"; {n(up)} round(s) could not be priced (every draft verified)" if up > 0 else ""))
    return L


def dsv41_cost_model_rows(m):
    """Leg PS2: where the round-cost table above comes from and how fresh it is (src/dsv41/gpu/dspark/metrics.rs
    render_model; the model is src/dsv41/gpu/dspark/roundcost.rs)."""
    L = []
    src = [dict(k[1]).get("source", "?") for k in m if k[0] == "velogb10_dsv41_spec_cost_source"]
    if not src:
        return L
    served = val(m, "velogb10_dsv41_spec_cost_served_rounds_total")
    age = val(m, "velogb10_dsv41_spec_cost_age_seconds")
    boot_age = val(m, "velogb10_dsv41_spec_cost_boot_age_seconds")
    what = {"served": f"refreshed from {n(served)} served round(s), last {age:.0} s ago",
            "boot": "this boot's calibration (no served round yet)",
            "boot_cache": "a calibration cached by an earlier boot of the same binary / pack / TP size / lanes / nodes / flags (no served round yet)"}.get(src[0], src[0])
    L.append(f"  round-cost table source: {what}; boot calibration measured {boot_age / 60:.0} min ago")
    L.append(f"    speed factor {val(m, 'velogb10_dsv41_spec_cost_speed_factor'):.3} (served walls over the fitted shape; 1 = as at boot), "
             f"{n(val(m, 'velogb10_dsv41_spec_cost_shifts_total'))} level shift(s), {n(val(m, 'velogb10_dsv41_spec_cost_outliers_total'))} outlier round(s); "
             f"host gap between rounds {val(m, 'velogb10_dsv41_spec_cost_gap_ms'):.2} ms")
    fit = {}
    for (nm, lb), x in m.items():
        if nm == "velogb10_dsv41_spec_cost_fit_ms":
            d = dict(lb)
            fit.setdefault(d.get("lanes", "?"), {})[d.get("part", "?")] = x
    if fit:
        L.append("    fitted fixed + per-draft cost: " + "; ".join(
            f"{ln} request(s) {fit[ln].get('fixed', 0):.1f} + {fit[ln].get('per_draft', 0):.1f} ms x drafts" for ln in sorted(fit, key=int)))
    return L


def dsv41_fw_rows(m):
    """DeepSeek-V4.1 TP engine paths (leg FW: src/dsv41/tp/metrics.rs): lanes, PK1's indexer decode plan, CED prefill
    rows, CUDA graphs. Empty unless a dsv41 TP comm attached in the process."""
    L = []
    if not any(k[0] == "velogb10_dsv41_fw_calls_total" for k in m):
        return L
    v = lambda name, **lb: val(m, "velogb10_dsv41_fw_" + name, **lb)
    calls, mixed, rows = v("calls_total"), v("mixed_lane_calls_total"), v("rows_total")
    L.append("")
    L.append("DEEPSEEK-V4.1 TP ENGINE (lanes / CED / graphs):")
    L.append(f"  forward calls: {n(calls)} ({n(rows)} rows); {n(mixed)} carried two or more lanes in one call"
             + ("" if mixed > 0 else " (no batched call yet: one request at a time, or lanes never overlapped)"))
    plan = {dict(k[1]).get("r", "?"): x for k, x in m.items() if k[0] == "velogb10_dsv41_fw_idx_decode_plan_total"}
    tot = sum(plan.values())
    if tot > 0:
        L.append(f"  index layers on PK1's decode plan: {n(tot)} (rows per block: " + ", ".join(f"r{r} {n(x)}" for r, x in sorted(plan.items())) +
                 f"; {n(v('idx_chunked_total'))} with the keys cut into chunks)")
    ced, ex = v("ced_layer_rows_total"), v("exact_layer_rows_total")
    if ex > 0:
        L.append(f"  prefill: {n(ced)} layer-rows computed for {n(ex)} an every-layer prefill would compute ({pct(ced, ex)}; CED saves the rest exactly)")
    cap, eager, rep = v("graph_captures_total"), v("graph_eager_total"), val(m, "velogb10_dsv41_tp_graph_replays_total")
    if cap > 0 or eager > 0:
        L.append(f"  CUDA graphs: {n(cap)} call shape(s) captured, {n(rep)} replay(s); {n(eager)} decode / verify call(s) ran eagerly (first call of a shape, or not graphable)")
        if any(k[0] == "velogb10_dsv41_fw_graph_shapes_seen" for k in m):
            ev = v("graph_evictions_total")
            km = [dict(k[1]).get("mode", "?") for k in m if k[0] == "velogb10_dsv41_fw_graph_keys"]
            tot = cap + eager + rep
            L.append(f"    graph keys: {km[0] if km else '?'} ({'per total rows; the lanes split is data' if km and km[0] == 'rows' else 'per (lane, rows) vector'}); "
                     f"{100 * rep / tot:.1f} % of graphable calls replayed" if tot > 0 else "    graph keys: no graphable call yet")
            L.append(f"    {n(v('graph_shapes_seen'))} distinct call shape(s) seen; {n(ev)} captured graph(s) destroyed by the 64-graph cap"
                     + (" (each re-captured on its next call: per-request row counts make more shapes)" if ev > 0 else ""))
    else:
        L.append("  CUDA graphs: OFF (--dsv41-tp-graphs off) or no decode / verify call yet")
    # leg PF1: programmatic dependent launch and the head over a round's lanes
    if any(k[0] == "velogb10_dsv41_xchg_pdl" for k in m):   # leg PX1
        xb = val(m, "velogb10_dsv41_xchg_pdl_built_total")
        wd = int(val(m, "velogb10_dsv41_tp_world")) or 2   # leg P4: the real world size (the line said "world 2" at world 4)
        L.append(f"  exchanges (world {wd}): " + (f"the dsv41 PDL pair, {n(xb)} exchange(s) built on it" if val(m, "velogb10_dsv41_xchg_pdl") > 0
                                              else "the shared K1 / K2 pair (--dsv41-xchg legacy, or not the dsv41 pair at this world)"))
    pdl = val(m, "velogb10_dsv41_pdl_launches_total")
    L.append(f"  programmatic dependent launch: {n(pdl)} launch(es) built with it" + ("" if pdl > 0 else " (OFF: --dsv41-pdl off, or no forward yet)"))
    hl, hs = v("head_launches_total"), v("head_segments_total")
    if hs > 0:
        L.append(f"  round head: {n(hl)} launch(es) for {n(hs)} lane segment(s)" + (" (lanes shared a head launch)" if hs > hl else " (one launch per lane: single-lane rounds, or lanes not consecutive)"))
    # leg PD1: the dense decode path (grouped launches, producer-side rotation) and the paced L2 prefetch
    if any(k[0] == "velogb10_dsv41_fw_pd1_dense_group" for k in m):
        g, r, pf = v("pd1_dense_group"), v("pd1_dense_rot"), v("pd1_l2pf")
        pc = v("pd1_calls_total")
        L.append(f"  dense decode path (PD1): grouped launches {'ON' if g else 'OFF (--dsv41-dense-group off)'}, input rotation in the producing kernel "
                 f"{'ON' if r else 'OFF (--dsv41-dense-rot off)'}; {n(pc)} decode / verify call(s) served on it"
                 + ("" if pc > 0 or not g else " (none yet: no decode call, or the pack's dense linears are not EXL3 integer widths)"))
        if pf:
            pl, pb, pcalls = v("pd1_l2pf_launches_total"), v("pd1_l2pf_bytes_total"), v("pd1_l2pf_calls_total")
            L.append(f"  paced L2 prefetch: ON; {n(pl)} launch(es) in {n(pcalls)} call(s), {pb / 1e9:.1f} GB asked into L2"
                     + ("" if pl > 0 else " (ENGAGED: NO: no decode call has run it yet)"))
        else:
            L.append("  paced L2 prefetch: OFF (--dsv41-l2-prefetch off, the default)")
    # leg PP1: the prefill paths (same bits as the paths they replace)
    if any(k[0] == "velogb10_dsv41_pf_router_wide_total" for k in m) and ex > 0:
        rw, gr = val(m, "velogb10_dsv41_pf_router_wide_total"), val(m, "velogb10_dsv41_pf_gemv_rows_total")
        L.append(f"  prefill (PP1): {n(rw)} router call(s) in one wide launch" + ("" if rw > 0 else " (ENGAGED: NO: --dsv41-pf-router-wide off, or --dsv41-router-tf32 on)")
                 + f"; {n(gr)} plain GEMV call(s) spread along the rows" + ("" if gr > 0 else " (ENGAGED: NO: --dsv41-pf-gemv-rows off, or the pack has no plain linear in the prefill)")
                 + "; Engram read-ahead: see the Engram block's 'read ahead' count (--dsv41-pf-engram-ahead)")
    # leg PA1: the attention sublayer's fused helpers, attention prologue / merge launch, hc side stream
    if any(k[0] == "velogb10_dsv41_fw_pa1_qkv_fuse" for k in m):
        fu, wp, mg, sd = v("pa1_qkv_fuse"), v("pa1_attn_wpf"), v("pa1_attn_merge"), v("pa1_hc_side")
        L.append(f"  attention sublayer (PA1): q / kv norm one launch + q RoPE in the attention {'ON' if fu else 'OFF (--dsv41-qkv-fuse off)'} "
                 f"({n(v('pa1_qkv_launches_total'))} fused launch(es), {n(v('pa1_qrope_launches_total'))} attention launch(es) with the RoPE built); "
                 f"window keys into L2 before the wait {'ON' if wp else 'OFF (--dsv41-attn-wpf off)'} ({n(v('pa1_wpf_launches_total'))}); "
                 f"per-head merge launch {'ON' if mg else 'OFF (--dsv41-attn-merge off)'} ({n(v('pa1_merge_launches_total'))})")
        L.append("  attention split rule: " + ("RECIPE (--dsv41-attn-split recipe: opt-in, different bits from exact; "
                 + n(v("pa1_attn_recipe_launches_total")) + " launch(es) built)" if v("pa1_attn_rule") else "exact (default)"))
        L.append(f"  hc mix on a side stream (PA1, TP1 / TP2 / TP4): {'ON' if sd else 'OFF (--dsv41-hc-side off, or TP8 and wider)'}; "
                 f"{n(v('pa1_hc_side_mixes_total'))} side mix(es) built" + (" (asked at TP8+: fell back to the reduce path)" if v("pa1_hc_side_fallback") else ""))
    return L


def dsv41_mem_rows(m):
    """DeepSeek-V4.1 TP memory (leg MEM1: src/dsv41/tp/mem.rs): what the head holds, how far the box is from the memory
    watchdog's floor, and whether the prefix pool had to give states back. Empty unless the series exist."""
    L = []
    if not any(k[0] == "velogb10_dsv41_mem_floor_bytes" for k in m):
        return L
    v = lambda name: val(m, "velogb10_dsv41_mem_" + name)
    parts = {dict(k[1]).get("part", "?"): x for k, x in m.items() if k[0] == "velogb10_dsv41_mem_bytes"}
    fl, av = v("floor_bytes"), v("available_bytes")
    L.append("")
    L.append("DEEPSEEK-V4.1 MEMORY (this rank):")
    if parts:
        w = parts.get("weights", 0)
        rest = sum(x for p, x in parts.items() if p != "weights")
        L.append(f"  holds: weights {w / 2**30:.1f} GiB + serving state {rest / 2**30:.2f} GiB (" +
                 ", ".join(f"{p} {x / 2**30:.2f}" for p, x in parts.items() if p != "weights" and x > 0) + ")")
    if fl > 0:
        L.append(f"  box: {av / 1e9:.1f} GB free, {(av - fl) / 1e9:+.1f} GB from the {fl / 1e9:.1f} GB memory-watchdog floor"
                 + ("  << BELOW THE FLOOR" if av < fl else ""))
    else:
        L.append("  box: the memory watchdog is OFF (--mem-watchdog-gb 0): nothing protects the box from running dry")
    L.append(f"  prefix pool allowance now: {v('pool_allowance_bytes') / 2**30:.2f} GiB; {n(v('pressure_evictions_total'))} kept states given back under memory pressure "
             f"in {n(v('pressure_events_total'))} events; {n(v('trims_total'))} trims; {int(v('graphs_live'))} captured graphs kept")
    if v("pressure_events_total") > 0:
        L.append("    the box ran short of memory at least once: the prefix pool shrank to keep the server above the floor (fewer prompt-cache hits)")
    return L


def dsv41_engram_rows(m):
    """The DeepSeek-V4.1 Engram row runtime rows (empty when no Engram runtime ran). Shared by the generic report and the DeepSeek-V4.1 one."""
    L = []
    # DSV41 EIO: the DeepSeek-V4.1 Engram row runtime (rows read from the NVMe per rank, pinned row cache, draft
    # prefetch, prompt read-ahead). Absent = no Engram runtime on this server.
    eng_steps = by_label(m, "velogb10_dsv41_engram_steps_total", "channel")
    if eng_steps:
        hit, miss = val(m, "velogb10_dsv41_engram_rows_total", outcome="hit"), val(m, "velogb10_dsv41_engram_rows_total", outcome="miss")
        st, to = val(m, "velogb10_dsv41_engram_stalls_total"), val(m, "velogb10_dsv41_engram_timeouts_total")
        L.append("")
        L.append("DEEPSEEK-V4.1 ENGRAM ROWS (this rank's heads, from the NVMe):")
        L.append(f"  {n(eng_steps.get('decode', 0))} decode round(s), {n(eng_steps.get('prefill', 0))} prefill chunk(s); rows {pct(hit, hit + miss)} from the "
                 f"row cache ({n(hit)} hit, {n(miss)} read, {n(val(m, 'velogb10_dsv41_engram_rows_total', outcome='dead'))} image positions skipped); "
                 f"read ahead: {n(val(m, 'velogb10_dsv41_engram_warm_rows_total', outcome='read'))} row(s)")
        if st > 0 or to > 0:
            L.append(f"  LATE ROWS: {n(st)} round(s) waited for Engram rows ({val(m, 'velogb10_dsv41_engram_stall_microseconds_total') / 1000:.1f} ms in total); "
                     f"{n(to)} bounded wait(s) timed out (those rounds failed)")
        else:
            L.append("  no round waited for an Engram row")
        er = val(m, "velogb10_dsv41_engram_read_errors_total")
        if er > 0:
            L.append(f"  READ ERRORS: {n(er)} row read(s) failed (their requests failed; check the NVMe)")
    return L

def dsv41_forward_rows(m):
    """The DeepSeek-V4.1 per-layer forward rows: which backend served each width (empty when no dsv41 forward ran)."""
    L = []
    # DSV41 SL: the DeepSeek-V4.1 per-layer forward's calls by path, weight backend, kernel entry, row bucket and the
    # class the backend served (velogb10_dsv41_fwd_calls_total). Absent = no dsv41 forward ran in this process.
    fwd = {}
    for (name, labels), v in m.items():
        if name == "velogb10_dsv41_fwd_calls_total":
            d = dict(labels)
            key = (d.get("backend", "-"), d.get("rows", "?"), d.get("class", "-"))
            fwd[key] = fwd.get(key, 0.0) + v
    if fwd:
        L.append("")
        L.append("DEEPSEEK-V4.1 FORWARD (which backend served the weights, by verify width):")
        for (b, rows, cls), v in sorted(fwd.items()):
            who = "weight-free kernels (mixers, caches, indexer, attention)" if b == "-" else f"{b} weights, class {cls}"
            L.append(f"  rows {rows:>5}: {n(v)} call(s) on {who}")
        gen = sum(v for (b, _, cls), v in fwd.items() if b != "-" and cls == "generic")
        if gen > 0:
            L.append(f"  {n(gen)} weight call(s) ran on a backend's GENERIC entry (no class kernel for that width yet: slower, counted)")
        else:
            L.append("  0 weight calls ran on a backend's generic fallback entry (every width served by a specialized kernel)")
    return L

def dsv41_state_rows(m):
    """UXF (RV1 F3, F4, F5): what a DeepSeek-V4.1 server did NOT do for a request, in plain words, from its own counters:
    top-k / top-p cut, thinking budget, penalties, fatal TP ops, boot memory-fit caps, prefill yields, PV1's decode paths.
    Kept apart from dsv41_rows so a branch that adds a memory section merges cleanly."""
    L = []
    v = lambda name, **lb: val(m, "velogb10_dsv41_" + name, **lb)
    has = lambda name: ("velogb10_dsv41_" + name, ()) in m
    L.append("")
    L.append("DEEPSEEK-V4.1 WHAT WAS SKIPPED, CAPPED OR LIMITED:")
    bad = False
    if v("tp_serve_fatal_total") > 0:
        bad = True
        L.append(f"  *** {n(v('tp_serve_fatal_total'))} TP OP(S) FAILED AND STOPPED THE FORWARD: the ranks can no longer be trusted to agree; the server answers errors until it is restarted ***")
    cu, cc = v("sampler_cut_unapplied_total"), v("sampler_cut_capped_total")
    if cu > 0:
        bad = True
        L.append(f"  top_p / top_k: {n(cu)} request(s) sent them and were served WITHOUT the cut (the sampled distribution was wider than asked)")
    if v("sampler_cut_rows_total") > 0:
        L.append(f"  top_p / top_k applied on {n(v('sampler_cut_rows_total'))} target row(s); {n(v('sampler_cut_wide_total'))} call(s) needed the wide candidate list")
    if cc > 0:
        bad = True
        L.append(f"  top_p / top_k CAPPED: {n(cc)} row(s) had a nucleus wider than 2,048 tokens; only the 2,048 best were kept (a request that asked for a wider top_p got a slightly narrower one)")
    bu = v("think_budget_unenforced_total")
    if bu > 0:
        bad = True
        L.append(f"  thinking_token_budget: {n(bu)} request(s) sent one and it was NOT enforced (the reasoning ran to the model's own end)")
    pu = v("sampler_penalty_unapplied_total")
    if pu > 0:
        bad = True
        L.append(f"  penalties: {n(pu)} request(s) sent presence / frequency / repetition penalties and were served WITHOUT them")
    if has("load_fit_caps"):
        caps = int(v("load_fit_caps"))
        la, ls = int(v("load_fit_lanes_asked")), int(v("load_fit_lanes_served"))
        ca, cs = int(v("load_fit_ctx_asked")), int(v("load_fit_ctx_served"))
        used, bud = v("load_fit_used_bytes") / 2**30, v("load_fit_budget_bytes") / 2**30
        if caps > 0:
            bad = True
            L.append(f"  MEMORY FIT: the asked posture did not fit this box; {caps} setting(s) were lowered at boot: lanes {la} -> {ls}, context {ca:,} -> {cs:,} "
                     f"(planned use {used:.1f} of {bud:.1f} GiB); the boot log names which")
        elif bud > 0:
            L.append(f"  memory fit: the asked posture fits (lanes {ls}, context {cs:,}; planned use {used:.1f} of {bud:.1f} GiB)")
    y = v("prefill_yields_total")
    if y > 0:
        L.append(f"  prefill yields: {n(y)} prefill chunk(s) were followed by a decode round of the other lanes (long prompts do not freeze them)")
    # PV1's decode paths (host launch counts: a CUDA-graph replay re-runs a captured launch without counting)
    mc, mch, mss = v("moe_dec_calls_total"), v("moe_dec_chain_calls_total"), v("moe_dec_short_scratch_total")
    ac, ag = v("attn_dec_calls_total"), v("attn_dec_grouped_calls_total")
    if has("moe_dec_calls_total"):
        L.append(f"  decode MoE (PV1, 1-4 rows): {n(mc)} launch(es) built, {n(mch)} with the K-split scratch; decode attention: {n(ac)} launch(es), {n(ag)} with split groups "
                 f"(counted when a launch is built, so CUDA-graph replays are not in these numbers)")
        if mss > 0:
            bad = True
            L.append(f"  *** {n(mss)} decode MoE call(s) were owed the K-split scratch and ran the one-pass path because the caller's scratch was short (same bits, without PV1's gain) ***")
        if mc == 0 and ac == 0:
            L.append("  (no decode launch has been built yet: send a request, then run this again)")
    # PM1: which decode MoE structure served (labelled per gate/up entry: d41x_mc_* short CTAs, d41x_md_* persistent)
    ent = by_label(m, "velogb10_dsv41_moe_dec_entry_calls_total", "entry")
    if ent:
        short = sum(c for e, c in ent.items() if e.startswith("d41x_mc_"))
        pers = sum(c for e, c in ent.items() if e.startswith("d41x_md_"))
        L.append(f"  decode MoE structure (PM1, --dsv41-moe-cta): {n(short)} launch(es) on the short-CTA kernels (d41x_mc_*, ready flags), {n(pers)} on the persistent kernels (d41x_md_*)")
        css = v("moe_cp_short_scratch_total")
        if css > 0:
            bad = True
            L.append(f"  *** {n(css)} decode MoE call(s) the short-CTA path would serve ran the persistent kernels because the caller's scratch was short (same bits, slower) ***")
    if not bad:
        L.append("  nothing was skipped, capped or lowered on this boot so far")
    return L


def heat_rows(m):
    """NIGHT1 heat: the thermal governor and how the host waits for the GPU (both decide whether a GB10 box overheats
    under sustained prefill), the TP proxies' idle tier, and (leg HT1) a DeepSeek-V4.1 TP server's idle tier and Engram
    readers. Shared by the generic report and the DeepSeek-V4.1 one (TP4B D6: a DSV4.1 server printed no heat at all)."""
    L = []
    if ("velogb10_thermal_guard", ()) in m or any(name == "velogb10_gpu_wait_policy" for (name, _) in m):
        L.append("")
        L.append("HEAT (this box)")
        if ("velogb10_thermal_guard", ()) in m:
            th = {dict(lb).get("which"): v for (name, lb), v in m.items() if name == "velogb10_thermal_celsius"}
            if val(m, "velogb10_thermal_guard") > 0:
                held, paced = val(m, "velogb10_thermal_held_seconds_total"), val(m, "velogb10_thermal_paced_seconds_total")
                L.append(f"  thermal guard ON: board now {th.get('now', 0):.1f} C, peak {th.get('peak', 0):.1f} C since boot; "
                         f"pacing from {th.get('soft', 0):.0f} C, holding at {th.get('hard', 0):.0f} C")
                L.append(f"  GPU work paused for heat: {paced:.1f} s paced ({n(val(m, 'velogb10_thermal_paced_total'))} pauses), "
                         f"{held:.1f} s held at the hard limit ({n(val(m, 'velogb10_thermal_holds_total'))} holds)"
                         + ("" if paced + held > 0 else " - never needed so far"))
            else:
                L.append("  thermal guard OFF (--thermal-guard off, or no GPU work yet): nothing slows GPU work down when the box "
                         "gets hot - a GB10 can power itself off under sustained prefill")
        pol = [dict(lb).get("policy") for (name, lb) in m if name == "velogb10_gpu_wait_policy"]
        if pol:
            fl = val(m, "velogb10_gpu_wait_context_flags")
            how = {"spin": "spin: every waiting thread holds a CPU core at 100% for the whole wait",
                   "hybrid": "hybrid: waits sleep until just before the GPU is due, then poll with short sleeps (no driver spin)",
                   "block": "block: every wait sleeps in the driver until the GPU signals (no spin; each wake-up costs ~0.6-1.4 ms)"}
            L.append(f"  host GPU waits (--gpu-wait): {how.get(pol[0], pol[0])}; driver scheduling flags {int(fl)}"
                     + (f"; {n(val(m, 'velogb10_gpu_wait_hybrid_waits_total'))} waits slept "
                        f"{val(m, 'velogb10_gpu_wait_slept_seconds_total'):.1f} s instead of spinning" if pol[0] == "hybrid" else ""))
        if val(m, "velogb10_tp_proxy_parks_total") > 0 or val(m, "velogb10_tp_proxy_parked_seconds_total") > 0:
            L.append(f"  TP proxy: parked {n(val(m, 'velogb10_tp_proxy_parks_total'))} times, "
                     f"{val(m, 'velogb10_tp_proxy_parked_seconds_total'):.1f} s instead of spinning a core")

    L += dsv41_idle_rows(m)
    return L


def dsv41_idle_rows(m):
    """Leg HT1: the DeepSeek-V4.1 TP server's idle tier (src/dsv41/tp/idle.rs) and its Engram readers' parking
    (src/dsv41/engram/io.rs), in plain words. Empty unless the server is a DSV4.1 TP head."""
    L = []
    if ("velogb10_dsv41_tp_idle_park_ms", ()) in m:
        ms = val(m, "velogb10_dsv41_tp_idle_park_ms")
        if ms > 0:
            now = "PARKED now (no collective, GPU kernel or spinning thread on any rank)" if val(m, "velogb10_dsv41_tp_idle_parked") > 0 else "awake now"
            parks = val(m, "velogb10_dsv41_tp_idle_parks_total")
            L.append(f"  idle tier ON (--dsv41-idle-park-ms {ms:.0f}): {now}; parked {n(parks)} time(s), "
                     f"{val(m, 'velogb10_dsv41_tp_idle_parked_seconds_total'):.1f} s in all, woken {n(val(m, 'velogb10_dsv41_tp_idle_wakes_total'))} time(s)"
                     + ("" if parks > 0 else " - NEVER parked yet (it parks after that long with no request)"))
            if val(m, "velogb10_dsv41_tp_idle_wakes_total") > 0:
                L.append(f"    first op after a wake took up to {1000 * val(m, 'velogb10_dsv41_tp_idle_first_op_seconds_max'):.1f} ms "
                         f"(last {1000 * val(m, 'velogb10_dsv41_tp_idle_first_op_seconds_last'):.1f} ms; includes the op's own GPU work)")
        else:
            L.append("  idle tier OFF (--dsv41-idle-park-ms 0): an idle server keeps every node's main thread, its RDMA proxy and a "
                     "GPU kernel spinning (a 250 ms heartbeat)")
    if ("velogb10_dsv41_tp_thermal_idle_gaps_total", ()) in m:
        g = val(m, "velogb10_dsv41_tp_thermal_idle_gaps_total")
        if g > 0:
            L.append(f"  thermal pacing billed GPU work only: {n(g)} idle gap(s) before a request's op ({val(m, 'velogb10_dsv41_tp_thermal_idle_gap_seconds_total'):.1f} s) "
                     f"were not counted as work, {val(m, 'velogb10_dsv41_tp_thermal_idle_credit_seconds_total'):.1f} s of pacing sleep not needed (this box)")
    if ("velogb10_dsv41_tp_round_wait_hybrid", ()) in m:
        if val(m, "velogb10_dsv41_tp_round_wait_hybrid") > 0:
            L.append("  inside a request each rank's main thread waits for the GPU with the hybrid wait (--dsv41-round-wait hybrid, opt-in): "
                     "the 'waits slept' count above includes it")
        else:
            L.append("  inside a request each rank's main thread spins in the driver while it waits for the GPU (--dsv41-round-wait spin, the "
                     "default; the --gpu-wait line above does not cover the DeepSeek-V4.1 path)")
    if any(k[0] == "velogb10_dsv41_engram_reader_parks_total" for k in m):
        L.append(f"  Engram readers (this box): parked {n(val(m, 'velogb10_dsv41_engram_reader_parks_total'))} time(s), "
                 f"{val(m, 'velogb10_dsv41_engram_reader_parked_seconds_total'):.1f} reader-seconds asleep instead of spinning; "
                 f"woken {n(val(m, 'velogb10_dsv41_engram_reader_wakes_total'))} time(s)")
    if L:
        L.insert(0, "  (DeepSeek-V4.1 TP: these counters are the head box's; each node paces and parks itself, its log says how often)")
    return L


def inkling2_rows(m):
    """The Inkling-2 serve shell, in plain words from its own counters (src/inkling2/serve/metrics.rs). T1 STUB: the chat-surface rows
    only; the integration leg adds the forward, speculation, lane-packing and prefix rows."""
    L = ["", "INKLING-2:"]
    v = lambda name, **lb: val(m, "velogb10_inkling2_" + name, **lb)
    if v("forward_not_built") > 0:
        L.append("  FORWARD: NOT BUILT. No model is behind this server: every chat request is answered with an error. Nothing here is a speed or quality number.")
    L.append(f"  requests: {n(v('requests_ok_total'))} finished, {n(v('requests_bad_total'))} refused as invalid (400), {n(v('requests_not_built_total'))} answered 'model not built', "
             f"{n(v('requests_engine_error_total'))} failed in the forward; streamed {n(v('streamed_requests_total'))}, not streamed {n(v('nonstreamed_requests_total'))}")
    L.append(f"  lanes: {int(v('lanes'))} (asked --max-batch {int(v('lanes_requested'))}); prefix cache {'ON' if v('prefix_cache') > 0 else 'off'}")
    eff = by_label(m, "velogb10_inkling2_reasoning_effort_requests_total", "how")
    L.append("  reasoning effort reached the template as: " + (", ".join(f"{k} {n(x)}" for k, x in sorted(eff.items()) if x) or "no request yet"))
    L.append(f"  tools: {n(v('tool_requests_total'))} requests offered tools, {n(v('tool_calls_total'))} calls returned, {n(v('tool_blocks_rejected_total'))} call payloads rejected by the catalogue (kept as text), "
             f"{n(v('tool_calls_over_limit_total'))} dropped by parallel_tool_calls false, {n(v('cut_calls_total'))} cut by the token budget (shown as text)")
    for lb, c in sorted(((dict(k[1]).get("reason"), x) for k, x in m.items() if k[0] == "velogb10_inkling2_tool_reject_reason_total"), key=lambda t: str(t[0])):
        if c > 0:
            L.append(f"    rejected because {lb}: {n(c)}")
    if v("forward_not_built") == 0:
        steps, rows = v("decode_steps_total"), v("decode_rows_total")
        world = int(v("tp_world")) if ("velogb10_inkling2_tp_world", ()) in m else 1
        L.append(f"  forward: GPU TP{world}{'' if world == 1 else ' (this box = rank 0, the head)'}; {n(v('prefill_tokens_total'))} prompt tokens prefilled; {n(steps)} decode steps, "
                 + (f"{rows / steps:.2f} lanes per step on average" if steps else "no decode step yet"))
        L.append("  expert activations: " + ("NATIVE NVFP4 / MXFP8 as the pack declares (--inkling2-expert-act native; differs from the 16-bit reference)"
                                           if v("expert_act_native") > 0 else "16-bit (default; the transformers reference's precision)"))
        L.append(f"  sampling: {n(v('gpu_sampled_tokens_total'))} tokens drawn on the GPU, {n(v('cpu_sampled_tokens_total'))} on the CPU (requests with top_p < 1 or top_k > 0)")
        if world > 1:
            ops, red = v("tp_ops_total"), v("tp_reduces_built_total")
            L.append(f"  tensor parallel: {world} GPUs; {n(ops)} model calls sent to the node(s), {n(red)} exchanges built ({v('tp_reduce_bytes_total') / 1e9:.2f} GB; "
                     "the decode graph replays its own exchanges without counting)" + ("" if ops else " -- NO call reached a node yet"))
        L.append(f"  long prompts: {n(v('interleaved_prefill_chunks_total'))} prefill chunks ran between other lanes' decode steps (they kept streaming)")
        if v("spec_on") > 0:
            rd, dr, ac, pr = v("spec_rounds_total"), v("spec_drafts_verified_total"), v("spec_drafts_accepted_total"), v("spec_plain_rows_total")
            L.append(f"  speculation: ON (bundled drafter): {n(rd)} drafting rounds, {n(dr)} drafts verified, {n(ac)} accepted "
                     + (f"({ac / dr:.0%} of verified; {(ac + rd) / rd:.2f} tokens per drafting round)" if rd and dr else "(no drafting round yet)")
                     + f"; {n(pr)} lane steps decoded plain while on (policy back-off / no room / CPU-sampled)")
            if v("spec_trunc_rounds_total") > 0:
                L.append(f"    top_p / top_k requests: {n(v('spec_trunc_rounds_total'))} drafting rounds accepted exactly against the truncated target (host accept)")
            if v("spec_kept_out_cpu_total") > 0:
                L.append(f"    kept out of speculation: {n(v('spec_kept_out_cpu_total'))} lane steps of requests with top_p < 1 or top_k > 0 (drawn on the CPU)")
            hist = by_label(m, "velogb10_inkling2_spec_accepted_rounds_total", "accepted")
            if rd:
                L.append("    accepted drafts per round: " + ", ".join(f"{k}: {n(x)}" for k, x in sorted(hist.items(), key=lambda t: int(t[0])) if x))
        else:
            L.append(f"  speculation: OFF ({n(v('spec_rounds_total'))} rounds; --mtp off, or the pack has no mtp.safetensors)")
        if v("prefix_cache") > 0:
            L.append(f"  prefix cache: {n(v('prefix_hits_total'))} requests resumed from a saved prompt end, {n(v('prefix_hit_tokens_total'))} prompt tokens not prefilled again"
                     + ("" if v("prefix_hits_total") else " (no request has continued an earlier prompt yet)"))
        else:
            L.append("  prefix cache: off (--prefix-cache off): every prompt is prefilled whole")
    if v("tool_choice_unenforced_total") > 0:
        L.append(f"  tool_choice required / named function: {n(v('tool_choice_unenforced_total'))} requests were served without the choice being forced")
    if ("velogb10_inkling2_vision_on", ()) in m:
        towers, hits = v("vision_towers_total"), v("vision_cache_hits_total")
        ms = v("vision_tower_ms_total")
        if v("vision_on") > 0:
            L.append(f"  vision: tower loaded, images up to {int(v('image_max_tokens')):,} LM tokens each; {n(v('image_parts_total'))} image parts served, "
                     f"{n(towers)} tower runs ({n(v('vision_tower_tokens_total'))} image tokens, "
                     + (f"{ms / towers:.0f} ms per image on average" if towers else "no tower run yet")
                     + f"), {n(hits)} images re-used from the tower cache; {n(v('image_rows_spliced_total'))} prompt rows filled with image features")
            if v("images_downscaled_total") > 0:
                L.append(f"    {n(v('images_downscaled_total'))} images were above --image-max-tokens and were DOWNSCALED to fit (the reference has no resize)")
            if v("images_bad_total") > 0:
                L.append(f"    {n(v('images_bad_total'))} image parts could not be decoded (answered 400 invalid_image)")
        else:
            L.append(f"  vision: the tower is NOT loaded (see the boot log); image requests are answered with an error ({n(v('image_parts_total'))} image parts seen)")
    elif v("image_parts_total") > 0:
        L.append(f"  images: {n(v('image_parts_total'))} image parts rendered into prompts (this build has no vision stage)")
    if v("headerless_text_total") > 0 or v("text_invocations_total") > 0:
        L.append(f"  odd replies: {n(v('headerless_text_total'))} began without a message header, {n(v('text_invocations_total'))} raw-text invocation blocks (kept as text)")
    return L


def dsv41_rows(m):
    """The DeepSeek-V4.1 serve shell, in plain words from its own counters (src/dsv41/serve/metrics.rs)."""
    L = ["", "DEEPSEEK-V4.1:"]
    v = lambda name, **lb: val(m, "velogb10_dsv41_" + name, **lb)
    if v("stub_forward") > 0:
        L.append("  FORWARD: the CPU STUB (--dsv41-stub-forward): canned replies, no model compute. Nothing here is a speed or quality number.")
    ok, bad, busy = v("requests_ok_total"), v("requests_bad_total"), v("requests_busy_total")
    L.append(f"  requests: {n(ok)} finished, {n(bad)} refused as invalid (400), {n(busy)} refused as busy (503 --max-waiting), "
             f"{n(v('requests_cancelled_total'))} cancelled (client left / stop string / backlog), {n(v('requests_engine_error_total'))} failed in the forward")
    L.append(f"  lanes: {int(v('lanes'))} (asked --max-batch {int(v('lanes_requested'))}); now {int(v('lanes_active'))} decoding, {int(v('waiting'))} waiting")
    s, ns = v("sampled_requests_total"), v("nonstreamed_requests_total")
    L.append(f"  sampling: {n(s)} sampled, {n(v('greedy_requests_total'))} greedy; seedless requests: {n(v('seedless_random_total'))} random seed, {n(v('seedless_prompt_hash_total'))} prompt-hash")
    L.append(f"  thinking: {n(v('thinking_requests_total'))} requests in thinking mode, {n(v('chat_mode_requests_total'))} in chat mode; "
             f"reasoning closed by the token budget {n(v('think_budget_forced_total'))} times")
    L.append(f"  tools: {n(v('tool_requests_total'))} requests offered tools, {n(v('tool_calls_total'))} calls returned, "
             f"{n(v('tool_blocks_rejected_total'))} DSML blocks rejected by the catalogue (kept as text), {n(v('tool_structural_breaks_total'))} structural breaks, "
             f"{n(v('tool_unterminated_total'))} unterminated")
    if any(k[0] == "velogb10_dsv41_vision_requests_total" for k in m):
        vr, vi, vt, vu = v("vision_requests_total"), v("vision_images_total"), v("vision_tokens_total"), v("vision_encode_us_total")
        enc_ms = vu / 1000.0
        per_img = f" ({enc_ms / vi:.1f} ms/image)" if vi > 0 else ""
        L.append(f"  vision (ViT + 3x3 aligner on rank 0): {n(vr)} multimodal request(s), {n(vi)} image(s) encoded ({n(vt)} image tokens, {enc_ms:.1f} ms GPU encode{per_img})")
    for lb, c in sorted(((dict(k[1]).get("reason"), x) for k, x in m.items() if k[0] == "velogb10_dsv41_tool_reject_reason_total"), key=lambda t: str(t[0])):
        if c > 0:
            L.append(f"    rejected because {lb}: {n(c)}")
    L.append(f"  stop strings: {n(v('stop_string_hits_total'))} requests ended by one; streams: {n(v('streamed_requests_total'))}, "
             f"keep-alive frames {n(v('sse_keepalives_total'))}, backlog cancels {n(v('stream_backlog_cancels_total'))}")
    if v("role_chunk_before_first_token_total") > 0:
        L.append(f"  !! {n(v('role_chunk_before_first_token_total'))} role chunks were sent before the first token (streaming-hygiene bug)")
    if v("logprobs_unsupported_total") > 0:
        L.append(f"  logprobs: {n(v('logprobs_unsupported_total'))} requests asked for them and were served logprobs: null (not computed by this server)")
    if v("response_format_unenforced_total") > 0:
        L.append(f"  response_format: {n(v('response_format_unenforced_total'))} requests had it described to the model in the prompt; it is NOT enforced by a token mask")
    if v("prefix_cache") > 0:
        h, mi = v("prefix_hits_total"), v("prefix_misses_total")
        L.append(f"  prefix cache: ON, pool {v('prefix_pool_bytes') / 2**30:.2f} of {v('prefix_pool_budget_bytes') / 2**30:.0f} GiB in {int(v('prefix_pool_entries'))} states; "
                 f"{n(h)} hits / {n(mi)} misses ({pct(h, h + mi)} of lookups), {n(v('prefix_reused_tokens_total'))} prompt tokens served from kept state, "
                 f"{n(v('prefix_snapshots_total'))} snapshots taken, {n(v('prefix_evictions_total'))} evicted")
        if h + mi > 0 and h == 0:
            L.append("    PREFIX CACHE ENGAGED 0 TIMES: no request resumed from a kept state (every prompt was read whole)")
        # leg CE2: kept states the forward could not resume a prompt from (graceful fallbacks; non-zero = a labelling defect to report)
        tr = v("prefix_tail_resumes_total")
        if tr > 0:
            L.append(f"    {n(tr)} request(s) resumed at (or a few tokens before) a kept prompt end: the same prompt again, its last tokens changed, or a short extension")
        rr, cf = v("prefix_resume_refused_total"), v("tp_resume_cold_fallback_total")
        if rr > 0:
            L.append(f"    !! {n(rr)} kept state(s) were offered for a prompt they cannot resume exactly; those requests fell back to "
                     f"another kept state or a cold prefill (served correctly, slower; a state-label defect: report it)")
        if cf > 0:
            L.append(f"    !! {n(cf)} resumed prefill(s) were redone cold by the TP head (last-resort fallback; served correctly, slower; report it)")
    else:
        L.append("  prefix cache: off (--prefix-cache not on); every prompt is read whole")
    if v("sink_reject") > 0:
        L.append(f"  sink rejection (opt-in, changes output bytes): ON; {n(v('sink_draft_zero_rejections_total'))} tokens rejected and replaced, "
                 f"{n(v('sink_fallthroughs_total'))} committed anyway after the alternates ran out; flagged rows: "
                 f"{n(val(m, 'velogb10_dsv41_sink_flagged_rows_total', row='draft_zero'))} draft-zero, {n(val(m, 'velogb10_dsv41_sink_flagged_rows_total', row='prompt'))} prompt")
    else:
        L.append("  sink rejection: off (opt-in --dsv41-sink-reject; DeepSeek's own engine has none)")
    # leg CE: prompt scoring (an exact prefill per request) and the opt-in replay's raised keeps; printed when the series exist
    if any(k[0] == "velogb10_dsv41_score_requests_total" for k in m):
        L.append(f"  prompt scoring (/v1/completions echo, max_tokens 0; exact prefill, every layer on every row): {n(v('score_requests_total'))} requests, "
                 f"{n(v('score_tokens_total'))} tokens scored")
    if v("ced_keep_raised_total") > 0:
        L.append(f"  CED replay: --dsv41-ced-keep below 134 was raised to 134 {n(v('ced_keep_raised_total'))} times")
    return L + dsv41_spec_rows(m) + dsv41_fw_rows(m) + dsv41_state_rows(m) + dsv41_mem_rows(m) + dsv41_engram_rows(m) + dsv41_forward_rows(m) + heat_rows(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="the server, e.g. http://HEAD-BOX-IP:9000 (a trailing /v1 is accepted)")
    src.add_argument("--file", help="a saved copy of the server's /metrics text")
    ap.add_argument("--window-seconds", type=int, default=0,
                    help="read the counters twice, this many seconds apart, and report only what happened in between")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for /metrics (default 5)")
    a = ap.parse_args()
    try:
        if a.file:
            if a.window_seconds:
                ap.error("--window-seconds needs --base")
            m, where = parse(open(a.file, encoding="utf-8", errors="replace").read()), a.file
        else:
            first = parse(fetch(a.base, a.timeout))
            if a.window_seconds > 0:
                print(f"counting for {a.window_seconds} s: send your requests now ...", file=sys.stderr)
                time.sleep(a.window_seconds)
                first = diff(first, parse(fetch(a.base, a.timeout)))
            m, where = first, a.base
    except OSError as e:
        print(f"cannot read the counters: {e}", file=sys.stderr)
        return 2
    if not any(k[0].startswith("velogb10_") for k in m):
        print("no velogb10_* series found: this is not a veloGB10 /metrics page", file=sys.stderr)
        return 2
    print("\n".join(report(m, where, a.window_seconds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
