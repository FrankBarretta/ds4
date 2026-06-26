#!/usr/bin/env python3
"""Fetch ONLY the router gate tensors of a DeepSeek V4 GGUF from Hugging Face
(via HTTP range reads) and build the geometric neighbour table (.ds4geo).

This avoids downloading the full ~81 GB model: only the tiny F16
`blk.*.ffn_gate_inp.weight` tensors (a few hundred MB) are read. CPU only.

Usage:
    python3 fetch_gate_geometry.py [-o geometry.ds4geo] [--degree 4] [--probe-only]
    python3 fetch_gate_geometry.py --repo R --file F.gguf ...
"""

import argparse
import sys

from huggingface_hub import HfFileSystem

import build_expert_geometry as beg

DEFAULT_REPO = "antirez/deepseek-v4-gguf"
DEFAULT_FILE = ("DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-"
                "chat-v2-imatrix.gguf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("-o", "--out", default="geometry.ds4geo")
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--token", default=None, help="HF token (else anonymous)")
    ap.add_argument("--probe-only", action="store_true",
                    help="only parse the header and report the model shape")
    args = ap.parse_args()

    path = f"{args.repo}/{args.file}"
    fs = HfFileSystem(token=args.token)

    try:
        size = fs.info(path)["size"]
        print(f"remote: hf://{path}  ({size / 1e9:.1f} GB total)", file=sys.stderr)
    except Exception as e:
        print(f"could not stat remote file: {e}", file=sys.stderr)
        return 2

    print("opening remote file (ranged reads) ...", file=sys.stderr)
    with fs.open(path, "rb") as f:
        if args.probe_only:
            # Parse header only: reuse the stream reader but stop after we have
            # the gate tensor metadata. read_gate_tensors_stream already reads
            # only the gate ranges, so for a pure probe we just report shapes.
            gate = beg.read_gate_tensors_stream(f)
            layers = sorted(gate)
            ne, nd = gate[layers[0]].shape
            print(f"PROBE: {len(layers)} MoE layers, {ne} experts, "
                  f"n_embd={nd}", file=sys.stderr)
            return 0

        gate = beg.read_gate_tensors_stream(f)

    print(f"fetched gate tensors for {len(gate)} layers; building neighbours "
          f"(degree={args.degree}) ...", file=sys.stderr)
    n_layer, n_expert, neigh, affin = beg.build_neighbors(gate, args.degree)
    beg.write_geometry(args.out, n_layer, n_expert, args.degree, neigh, affin)
    print(f"wrote {args.out}: {n_layer} layers x {n_expert} experts "
          f"x {args.degree} neighbours", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
