#!/usr/bin/env python3
"""Build a co-activation neighbour table (.ds4geo) from a real expert trace.

For each layer L and expert e, the neighbours are the experts that most often
get selected at layer L within the next `--window` tokens AFTER e is selected
(temporal co-activation) — exactly the "what will be needed soon" signal a
prefetcher wants. Output is the same .ds4geo format the C tools load, so it is a
drop-in alternative to the gate-direction geometry.

Honest use: train this on one slice of a trace and test on a held-out slice.
"""
import argparse, struct
from collections import defaultdict, Counter
import numpy as np

MAGIC = b"DS4GEO\x00\x00"
VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("-o", "--out", default="coact.ds4geo")
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--n_layer", type=int, default=43)
    ap.add_argument("--n_expert", type=int, default=256)
    a = ap.parse_args()
    NL, NE, D, W = a.n_layer, a.n_expert, a.degree, a.window

    # per-layer sequence of selected sets, in token order
    seq = defaultdict(list)
    for line in open(a.trace):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        lay = int(p[0])
        if lay >= NL:
            continue
        seq[lay].append([int(x) for x in p[1:]])

    C = [defaultdict(Counter) for _ in range(NL)]
    pairs = 0
    for L in range(NL):
        s = seq.get(L, [])
        for i in range(len(s)):
            for e in s[i]:
                for j in range(i + 1, min(i + 1 + W, len(s))):
                    for f in s[j]:
                        if f != e:
                            C[L][e][f] += 1
                            pairs += 1

    neigh = np.zeros((NL, NE, D), dtype=np.uint16)
    affin = np.zeros((NL, NE, D), dtype=np.float32)
    covered = 0
    for L in range(NL):
        for e in range(NE):
            top = C[L][e].most_common(D)
            if top:
                covered += 1
            for r in range(D):
                if r < len(top):
                    neigh[L, e, r] = top[r][0]
                    affin[L, e, r] = float(top[r][1])
                else:
                    neigh[L, e, r] = e          # pad with self -> no-op prefetch
                    affin[L, e, r] = -1.0

    with open(a.out, "wb") as fo:
        fo.write(MAGIC)
        fo.write(struct.pack("<IIII", VERSION, NL, NE, D))
        fo.write(neigh.astype("<u2").tobytes())
        fo.write(affin.astype("<f4").tobytes())
    print(f"wrote {a.out}: window={W} degree={D} pairs={pairs} "
          f"experts_with_data={covered}/{NL*NE}")


if __name__ == "__main__":
    main()
