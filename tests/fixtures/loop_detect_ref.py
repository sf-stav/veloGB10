#!/usr/bin/env python3
"""WP08 reference: the rival's streaming loop detector, VERBATIM logic, torch-free.

Source: vcruz305/exllamav3 @523ecd3, exllamav3/generator/loop_detect.py (LoopDetector), wired by
job.py:312-318 as LoopDetector(window_size, window_size // min_reps) and fed every emitted token
(job.py:1010-1013). Only the torch import / tensor branch of feed_many is dropped.

Run to regenerate the cross-check vectors that src/loop_detect.rs's unit test replays:

    python3 tests/fixtures/loop_detect_ref.py > tests/fixtures/loop_detect_vectors.txt

Each vector line is `<spec> | first=<i> count=<n> sum=<s> period=<p>`: the sequence is rebuilt
from <spec> with the splitmix64 generator below (mirrored in the Rust test), fed token by token,
and every index at which feed() returned True is summarised (first index, how many, their sum,
and the detector's period at the first detection). -1 = never.
"""
import heapq
import sys


class LoopDetector:
    def __init__(self, window_size: int = 1000, max_period: int = None):
        self.W = window_size
        self.max_period = min(max_period or window_size // 3, window_size // 2)
        self._buf = [None] * self.W
        self._total = 0
        self._streak = [0] * (self.max_period + 1)
        self._heap = []
        self._gen = [0] * (self.max_period + 1)
        for p in range(1, self.max_period + 1):
            heapq.heappush(self._heap, (self.W, p, 0))
        self._detected_period = None

    def _schedule(self, p, wake_time):
        self._gen[p] += 1
        heapq.heappush(self._heap, (wake_time, p, self._gen[p]))

    def _backlog_scan(self, p):
        t = self._total
        streak = 0
        for k in range(self.W - p):
            if self._buf[(t - 1 - k) % self.W] == self._buf[(t - 1 - k - p) % self.W]:
                streak += 1
            else:
                break
        return streak

    def feed(self, token) -> bool:
        pos = self._total % self.W
        self._buf[pos] = token
        self._total += 1
        if self._total < self.W:
            return False
        t = self._total
        detected = False
        while self._heap and self._heap[0][0] <= t:
            wake_time, p, gen = heapq.heappop(self._heap)
            if gen != self._gen[p]:
                continue
            if self._streak[p] > 0:
                if self._buf[(t - 1) % self.W] == self._buf[(t - 1 - p) % self.W]:
                    self._streak[p] += 1
                    if self._streak[p] >= self.W - p:
                        self._detected_period = p
                        detected = True
                    self._schedule(p, t + 1)
                else:
                    self._streak[p] = 0
                    if self._detected_period == p:
                        self._detected_period = None
                    self._schedule(p, t + self.W)
            else:
                streak = self._backlog_scan(p)
                self._streak[p] = streak
                if streak >= self.W - p:
                    self._detected_period = p
                    detected = True
                    self._schedule(p, t + 1)
                elif streak > 0:
                    self._schedule(p, t + 1)
                else:
                    self._schedule(p, t + self.W)
        return detected or self._detected_period is not None


MASK = (1 << 64) - 1


class Rng:
    """splitmix64 — mirrored bit-for-bit by the Rust test."""
    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n


def build(spec):
    """spec = 'W,R,seed;seg;seg...' with seg = R:n:v (random) | P:n:p:v (period-p pattern
    repeated for n tokens) | Q:n:p:v:k (like P, every k-th token replaced by a random one)."""
    head, *segs = spec.split(";")
    w, reps, seed = (int(x) for x in head.split(","))
    rng = Rng(seed)
    toks = []
    for s in segs:
        f = s.split(":")
        kind, n = f[0], int(f[1])
        if kind == "R":
            v = int(f[2])
            toks += [rng.below(v) for _ in range(n)]
        elif kind in ("P", "Q"):
            p, v = int(f[2]), int(f[3])
            pat = [rng.below(v) for _ in range(p)]
            k = int(f[4]) if kind == "Q" else 0
            for i in range(n):
                if k and i % k == k - 1:
                    toks.append(rng.below(v))
                else:
                    toks.append(pat[i % p])
        else:
            raise ValueError(s)
    return w, reps, toks


CASES = [
    # periods 1, 2, 7, 100 detect; 101 never (W=300, max period = 300 // 3 = 100)
    "300,3,1;R:50:50000;P:800:1:50000",
    "300,3,2;R:50:50000;P:800:2:50000",
    "300,3,3;R:50:50000;P:800:7:50000",
    "300,3,4;R:50:50000;P:900:100:50000",
    "300,3,5;R:50:50000;P:1200:101:50000",
    "300,3,6;P:1000:3:50000",
    "300,3,7;R:1:50000;P:700:50:50000",
    "300,3,8;R:299:50000;P:700:99:50000",
    # one mismatch resets: loop, a single random token, loop again (sleep schedule is exercised)
    "300,3,9;R:40:50000;P:250:5:50000;R:1:50000;P:600:5:50000",
    "300,3,10;P:280:4:50000;R:1:50000;P:280:4:50000;R:1:50000;P:600:4:50000",
    "300,3,11;R:10:50000;P:305:9:50000;R:1:50000;P:800:9:50000",
    # perturbed loops (every k-th token random) never satisfy the whole-window condition
    "300,3,12;Q:3000:6:50000:97",
    "300,3,13;Q:3000:40:50000:250",
    # 10K random tokens never fire (large and small vocabularies)
    "300,3,14;R:10000:1000",
    "300,3,15;R:10000:248320",
    # tiny vocabularies: accidental periodic windows exercise the backlog scan heavily
    "300,3,16;R:5000:2",
    "300,3,17;R:5000:3",
    "40,3,18;R:5000:2",
    "40,3,19;R:3000:4",
    # period changes mid-stream (the detector must re-detect the new period)
    "300,3,20;P:500:7:50000;P:900:13:50000",
    "300,3,21;P:400:2:50000;R:30:50000;P:900:31:50000",
    # other window / reps settings (the CLI allows them)
    "100,2,22;R:60:50000;P:400:50:50000",
    "100,2,23;R:60:50000;P:400:51:50000",
    "64,4,24;R:5:50000;P:300:16:50000;R:1:50000;P:300:3:50000",
    "2048,3,25;R:100:50000;P:4000:600:50000",
    "301,3,26;R:77:50000;P:1500:100:50000",
]


def summarize(spec):
    w, reps, toks = build(spec)
    d = LoopDetector(w, w // reps)
    first, count, total, period = -1, 0, 0, -1
    for i, t in enumerate(toks):
        if d.feed(t):
            if first < 0:
                first, period = i, (d._detected_period if d._detected_period is not None else -1)
            count += 1
            total += i
    return f"{spec} | first={first} count={count} sum={total} period={period}"


if __name__ == "__main__":
    for c in CASES:
        print(summarize(c))
    sys.stdout.flush()
