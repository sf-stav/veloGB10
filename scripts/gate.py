#!/usr/bin/env python3
"""The gates, run as HYPOTHESIS TESTS instead of anecdotes.

    ./scripts/gate.py --model-dir /path/to/9b-nvfp4-mixed
    ./scripts/gate.py --model-dir ... --quick        # smaller SPRT bound, for a fast pre-commit check

Two failures this project has actually shipped motivate every line here.

1. A GATE THAT FAILS AS A COIN TOSS IS NOT A GATE. The split-K losslessness bug passed most runs: a
   1-ulp difference rarely flips an argmax, so a single green run said almost nothing. Running a test
   once is a sample size of one. Wald's SPRT (1945) answers "how many clean runs is green": to
   separate "never fails" from "fails 30% of the time" at 1% error rates takes ~13-15 consecutive
   passes -- and it stops the instant one fails. So the losslessness gate is a LOOP with a stopping
   rule, over RANDOMIZED (ctx, offset, depth) draws, not one fixed invocation.

2. A COMPARISON THAT CAN SILENTLY COMPARE NOTHING FAILS OPEN. A fuzz harness once grepped for a string
   the binary never prints, compared empty to empty, and reported IDENTICAL. A "smoke test" grepped for
   error strings the failure didn't contain and reported OK on a binary that never started. A unit test
   looked for a file the sharded models don't have and "passed" in 0.00s having loaded nothing. So:
   every check here asserts that it EXTRACTED something before it judges it. No silent skips, ever.

It also keeps a CSV of every run's metrics and flags a >3-sigma excursion against that history
(Shewhart control chart). Slow drifts -- a stale PTX quietly serving old kernels at old speeds -- are
invisible to a pass/fail threshold and obvious on a chart.

Exit status: 0 = green, 1 = a gate failed, 2 = the harness could not measure (which is NOT green).
"""
import argparse, csv, math, os, random, re, statistics, subprocess, sys, time

BIN = "./target/release/gb10_inference"
HIST = "scripts/gate_history.csv"


def run(args, timeout=3000):
    """Run the binary and return its combined output. Never let a crash look like a pass."""
    p = subprocess.run([BIN] + args, capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr, p.returncode


def extract(pattern, text, what):
    """Pull a value out of the output, and FAIL LOUDLY if it is not there.

    This is the whole point. A regex that matches nothing must never be read as 'no problem found'.
    """
    m = re.search(pattern, text)
    if not m:
        print(f"\n  HARNESS FAILURE: could not find {what} in the output.", file=sys.stderr)
        print(f"  Looked for: {pattern!r}", file=sys.stderr)
        print("  This is NOT a pass. The harness could not measure what it claims to measure.",
              file=sys.stderr)
        print("  --- last 25 lines of output ---", file=sys.stderr)
        for line in text.strip().splitlines()[-25:]:
            print("   ", line, file=sys.stderr)
        sys.exit(2)
    return m.group(1) if m.groups() else m.group(0)


def sprt_bound(p0, p1, alpha, beta):
    """Consecutive clean runs needed to accept H0 (p_fail <= p0) against H1 (p_fail >= p1).

    With zero observed failures the Wald log-likelihood ratio walks by log((1-p1)/(1-p0)) per clean
    run; we accept H0 when it crosses log(beta/(1-alpha)).
    """
    step = math.log((1 - p1) / (1 - p0))          # negative
    lower = math.log(beta / (1 - alpha))          # negative
    return int(math.ceil(lower / step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", default=None, help="corpus to slice contexts from")
    ap.add_argument("--quick", action="store_true", help="p1=0.5 instead of 0.3 (fewer runs)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    if not os.path.exists(BIN):
        print(f"no binary at {BIN} — run: cargo build --release", file=sys.stderr); sys.exit(2)

    seed = a.seed if a.seed is not None else int(time.time())
    rng = random.Random(seed)

    # H1: a failure rate of 30% (or 50% with --quick) is what "fails as a coin toss" means in practice.
    p0, p1, alpha, beta = 0.001, (0.5 if a.quick else 0.3), 0.01, 0.01
    n_needed = sprt_bound(p0, p1, alpha, beta)

    corpus = a.prompt_file
    if not corpus:
        for c in ("ppl_holdout.txt", os.environ.get("GB10_PPL_HOLDOUT", ""), "AGENTS.md"):
            if c and os.path.exists(c): corpus = c; break
    if not corpus:
        raise SystemExit("no perplexity corpus found — pass --prompt-file, set GB10_PPL_HOLDOUT, or run from the repo root")
    text = open(corpus).read()
    if len(text) < 60000:
        text = text * (60000 // max(len(text), 1) + 1)

    print(f"GATE  model={os.path.basename(a.model_dir)}  seed={seed}")
    print(f"  SPRT: to reject 'fails >= {p1:.0%} of the time' at alpha=beta={alpha:.0%}, "
          f"need {n_needed} consecutive clean runs (stops early on the first failure).\n")

    metrics = {}
    failures = []

    # ---- Gate 1: greedy MTP losslessness, randomized. The offsets straddle the 256-key split-K
    # boundary on purpose: that is exactly where the shipped bug lived, and a fixed ctx never hit it.
    # ONE process for all draws: the model load is ~11 s and the gate itself is ~1 s, so spawning a
    # process per draw spent 97% of the wall clock re-reading a 6 GB artifact. A gate you won't wait
    # for is a gate you skip.
    print(f"[1/3] losslessness fuzz — {n_needed} randomized (ctx, offset, depth) draws, one process")
    out, rc = run(["--bench-verify", "--model-dir", a.model_dir, "--prompt", text[:32000],
                   "--draws", str(n_needed), "--seed", str(seed % (2**63)),
                   "--max-seq-len", "16384"])
    for line in out.splitlines():
        if line.strip().startswith("draw "):
            print("  " + line.strip())
    verdict = extract(r"RESULT: (LOSSLESS_OK|MISMATCH)", out, f"a losslessness verdict over {n_needed} draws")
    if verdict != "LOSSLESS_OK" or rc != 0:
        failures.append(f"losslessness: {verdict} over {n_needed} randomized draws (seed {seed}, rc={rc})")
    else:
        print(f"   => {n_needed}/{n_needed} clean. 'fails >= {p1:.0%}' rejected at {alpha:.0%}.\n")

    # ---- Gate 2: batch invariance (column 0 bit-identical for every N).
    print("[2/3] batch invariance (probe-binv)")
    out, rc = run(["--probe-binv", "--model-dir", a.model_dir])
    verdict = extract(r"\b(PASS|FAIL)\b", out, "a probe-binv verdict")
    print(f"   {verdict}")
    if verdict != "PASS" or rc != 0:
        failures.append(f"batch invariance: {verdict} (rc={rc})")

    # ---- Gate 3: end-to-end + ACCEPTANCE. The lossless gate does NOT prove draft quality: a broken
    # re-prime makes bad drafts, the verify rejects them, output stays correct, and the speedup is gone.
    print("\n[3/3] end-to-end MTP + acceptance (the draft-side canary)")
    out, rc = run(["--bench-mtp", "--model-dir", a.model_dir, "--depth", "2", "--max-new-tokens", "64"])
    verdict = extract(r"(LOSSLESS_OK|MISMATCH)", out, "an end-to-end verdict")
    acc = float(extract(r"acceptance rate:\s*([\d.]+)%", out, "an acceptance rate"))
    tps = float(extract(r"throughput: MTP\s+([\d.]+) tok/s", out, "an MTP throughput"))
    print(f"   {verdict}   acceptance {acc:.1f}%   {tps:.1f} tok/s")
    if verdict != "LOSSLESS_OK" or rc != 0:
        failures.append(f"end-to-end: {verdict} (rc={rc})")
    metrics = {"acceptance": acc, "mtp_tok_s": tps}

    # ---- Control chart: compare against this model's own history, not a hand-picked threshold.
    key = os.path.basename(a.model_dir.rstrip("/"))
    hist = []
    if os.path.exists(HIST):
        with open(HIST) as f:
            hist = [r for r in csv.DictReader(f) if r["model"] == key]
    # THE BASELINE MUST BE FROZEN.
    #
    # My first version recomputed mu/sigma from ALL history on every run — including the runs that had
    # already drifted. So the baseline slid along with the drift, sigma inflated, z shrank, and the
    # chart quietly normalised to the very failure it exists to catch. I tested it against a simulated
    # stale-PTX sag and CUSUM never fired, not once, over ten drifted runs.
    #
    # SPC's answer since Shewhart: establish limits from a STABLE BASELINE period and hold them fixed.
    # Here: the earliest `BASELINE_N` GREEN runs for this model. Everything after is judged against them.
    BASELINE_N = 8
    print("\n[chart] vs frozen baseline")
    green_hist = [r for r in hist if r.get("green") == "1"]
    if len(green_hist) < BASELINE_N:
        print(f"   {len(green_hist)}/{BASELINE_N} green runs for {key} — baseline not established yet. "
              f"(Run the gate on known-good builds to seed it.)")
    else:
        baseline = green_hist[:BASELINE_N]
        after = green_hist[BASELINE_N:]
        for name, val in metrics.items():
            base = [float(r[name]) for r in baseline if r.get(name)]
            past = [float(r[name]) for r in after if r.get(name)]   # judged, never absorbed
            mu = statistics.mean(base)

            # A SIGMA FLOOR, or the chart cries wolf. Two identical runs give sd = 0, and then any
            # third value at all is "infinitely many sigma" from the mean. A chart that fires on noise
            # gets ignored, which is the same as having no chart. Floor sigma at 0.5% of the mean --
            # below that, a change is not something we can distinguish from run-to-run jitter anyway.
            sd = max(statistics.pstdev(base), abs(mu) * 0.005, 1e-6)

            # SHEWHART: catches a big JUMP (>3 sigma) on this one run.
            z = (val - mu) / sd
            shewhart = abs(z) > 3

            # CUSUM: catches a small SUSTAINED drift that Shewhart is blind to by construction.
            # This is the one that matters for us: a stale PTX serving old kernels at old speeds is not
            # a 3-sigma jump, it is a persistent ~1-sigma sag that every single-run threshold waves
            # through. Tabular CUSUM with slack k=0.5 sigma and decision interval h=5 sigma detects a
            # sustained 1-sigma shift within ~10 runs at a low false-alarm rate.
            k, h = 0.5, 5.0
            sp = sm = 0.0
            for x in past + [val]:
                d = (x - mu) / sd
                sp = max(0.0, sp + d - k)
                sm = max(0.0, sm - d - k)
            cusum = max(sp, sm) > h

            flag = ("  <-- JUMP (>3 sigma)" if shewhart else
                    "  <-- DRIFT (CUSUM)" if cusum else "")
            print(f"   {name:12} {val:8.2f}   history {mu:.2f} +/- {sd:.2f}   "
                  f"z={z:+.1f}  cusum={max(sp, sm):.1f}/{h:.0f}{flag}")
            if shewhart or cusum:
                what = "jumped" if shewhart else "has been DRIFTING"
                failures.append(
                    f"control chart: {name} {what} — now {val:.2f}, history {mu:.2f}+/-{sd:.2f} "
                    f"(z={z:+.1f}, cusum={max(sp, sm):.1f}). Not necessarily a bug, but do not ship "
                    f"without explaining it. A stale PTX looks exactly like this.")

    # Record this run whatever the verdict — a chart with only the good runs on it is not a chart.
    new = not os.path.exists(HIST)
    with open(HIST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "model", "seed", "green", "acceptance", "mtp_tok_s"])
        if new: w.writeheader()
        w.writerow({"ts": int(time.time()), "model": key, "seed": seed,
                    "green": int(not failures), **{k: f"{v:.3f}" for k, v in metrics.items()}})

    print()
    if failures:
        print("GATE RED")
        for f_ in failures: print(f"  - {f_}")
        sys.exit(1)
    print(f"GATE GREEN  ({n_needed} clean losslessness draws, invariance PASS, "
          f"acceptance {metrics['acceptance']:.1f}%)")


if __name__ == "__main__":
    main()
