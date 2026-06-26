#!/usr/bin/env python3
"""Build the geometric expert-neighbour table (.ds4geo) from a DeepSeek V4 GGUF.

This is the OFFLINE half of the geometric expert-prefetch feature. It reads the
router gate matrices `blk.*.ffn_gate_inp.weight` (shape [n_embd, n_expert], F16)
from a GGUF, treats each expert's gate-direction row as a point in embedding
space, and writes the top-`degree` nearest experts (by cosine similarity) per
expert and layer. ds4 (phase-2 runtime, opt-in) loads this file to prefetch an
expert's geometric neighbours into the streaming VRAM cache.

It runs on CPU, needs no GPU, and does NOT modify the GGUF or any ds4 source.
Only the tiny F16 gate tensors are read (a few hundred MB), not the experts.

Requires: numpy.

Usage:
    python3 build_expert_geometry.py MODEL.gguf -o geometry.ds4geo [--degree 4]

The output format matches ds4_expert_geometry_save() / _load():
    magic "DS4GEO\\0" (8 bytes)
    uint32 version(=1), n_layer, n_expert, degree   (little-endian)
    uint16 neighbor[n_layer*n_expert*degree]
    float32 affinity[n_layer*n_expert*degree]
"""

import argparse
import re
import struct
import sys

import numpy as np

MAGIC = b"DS4GEO\x00\x00"  # 8 bytes, matches C DS4_EXPERT_GEOMETRY_MAGIC "DS4GEO\0"
VERSION = 1

GGUF_GATE_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")

# ---- minimal GGUF reader (stream based, reads only the gate tensors) -------

_GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
_GGML_F32 = 0
_GGML_F16 = 1

# GGUF metadata value type ids -> (struct fmt, size)
_GGUF_SIMPLE = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


def _read(fmt, f):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _read_str(f):
    (n,) = _read("<Q", f)
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, vtype):
    """Read and return a metadata value (arrays/unknown -> None, but consumed)."""
    if vtype in _GGUF_SIMPLE:
        fmt, _ = _GGUF_SIMPLE[vtype]
        return _read(fmt, f)[0]
    if vtype == 8:  # string
        return _read_str(f)
    if vtype == 9:  # array
        (elem_type,) = _read("<I", f)
        (count,) = _read("<Q", f)
        for _ in range(count):
            _read_value(f, elem_type)
        return None
    raise ValueError(f"unknown GGUF metadata value type {vtype}")


def read_gate_tensors_stream(f):
    """Parse a GGUF header from a seekable binary stream and return
    {layer_index: np.ndarray[n_expert, n_embd] float32} for the gate tensors.
    Only the gate tensor byte ranges are read from `f` (good for remote/ranged
    file objects)."""
    magic, _version = _read("<II", f)
    if magic != _GGUF_MAGIC:
        raise ValueError("not a GGUF file (bad magic)")
    n_tensors, n_kv = _read("<QQ", f)

    alignment = 32
    for _ in range(n_kv):
        key = _read_str(f)
        (vtype,) = _read("<I", f)
        val = _read_value(f, vtype)
        if key == "general.alignment" and isinstance(val, int):
            alignment = val

    tensors = []
    for _ in range(n_tensors):
        name = _read_str(f)
        (n_dims,) = _read("<I", f)
        dims = [_read("<Q", f)[0] for _ in range(n_dims)]
        (ttype,) = _read("<I", f)
        (offset,) = _read("<Q", f)
        tensors.append((name, dims, ttype, offset))

    pos = f.tell()
    data_start = (pos + alignment - 1) // alignment * alignment

    out = {}
    gate_tensors = [(n, d, t, o) for (n, d, t, o) in tensors if GGUF_GATE_RE.match(n)]
    if not gate_tensors:
        raise ValueError("no blk.*.ffn_gate_inp.weight tensors found")

    for i, (name, dims, ttype, offset) in enumerate(gate_tensors):
        if ttype not in (_GGML_F16, _GGML_F32):
            raise ValueError(f"{name}: expected F16/F32 gate, got ggml type {ttype}")
        n_embd, n_expert = int(dims[0]), int(dims[1])  # fastest axis = n_embd
        count = n_embd * n_expert
        f.seek(data_start + offset)
        if ttype == _GGML_F16:
            raw = np.frombuffer(f.read(count * 2), dtype=np.float16)
            arr = raw.astype(np.float32)
        else:
            arr = np.frombuffer(f.read(count * 4), dtype=np.float32).copy()
        layer = int(GGUF_GATE_RE.match(name).group(1))
        out[layer] = arr.reshape(n_expert, n_embd)  # rows = experts
        if (i % 8) == 0 or i + 1 == len(gate_tensors):
            print(f"  read gate {i + 1}/{len(gate_tensors)} "
                  f"(layer {layer}: {n_expert} experts x {n_embd})",
                  file=sys.stderr)
    return out


def load_gate_tensors(path):
    with open(path, "rb") as f:
        return read_gate_tensors_stream(f)


# ---- neighbour computation ------------------------------------------------

def build_neighbors(gate_by_layer, degree):
    layers = sorted(gate_by_layer)
    n_layer = len(layers)
    n_expert = gate_by_layer[layers[0]].shape[0]

    neigh = np.zeros((n_layer, n_expert, degree), dtype=np.uint16)
    affin = np.zeros((n_layer, n_expert, degree), dtype=np.float32)

    for li, layer in enumerate(layers):
        V = gate_by_layer[layer].astype(np.float32)
        if V.shape[0] != n_expert:
            raise ValueError("inconsistent expert count across layers")
        norm = np.linalg.norm(V, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Vn = V / norm
        sim = Vn @ Vn.T                 # [n_expert, n_expert] cosine
        np.fill_diagonal(sim, -2.0)     # exclude self
        keep = min(degree, n_expert - 1)
        idx = np.argpartition(-sim, keep - 1, axis=1)[:, :degree]
        rows = np.arange(n_expert)[:, None]
        order = np.argsort(-sim[rows, idx], axis=1)
        idx = idx[rows, order]
        neigh[li] = idx.astype(np.uint16)
        affin[li] = sim[rows, idx].astype(np.float32)
        print(f"  layer {layer}: mean top-1 cos = {affin[li, :, 0].mean():.3f}",
              file=sys.stderr)

    return n_layer, n_expert, neigh, affin


def write_geometry(path, n_layer, n_expert, degree, neigh, affin):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, n_layer, n_expert, degree))
        f.write(neigh.astype("<u2").tobytes())
        f.write(affin.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to the DeepSeek V4 GGUF")
    ap.add_argument("-o", "--out", default="geometry.ds4geo",
                    help="output .ds4geo path (default: geometry.ds4geo)")
    ap.add_argument("--degree", type=int, default=4,
                    help="neighbours stored per expert (default: 4)")
    args = ap.parse_args()

    if args.degree < 1 or args.degree > 64:
        ap.error("--degree must be in 1..64")

    print(f"reading gate tensors from {args.model} ...", file=sys.stderr)
    gate = load_gate_tensors(args.model)
    print(f"found {len(gate)} MoE layers", file=sys.stderr)

    n_layer, n_expert, neigh, affin = build_neighbors(gate, args.degree)
    write_geometry(args.out, n_layer, n_expert, args.degree, neigh, affin)
    print(f"wrote {args.out}: {n_layer} layers x {n_expert} experts "
          f"x {args.degree} neighbours", file=sys.stderr)


if __name__ == "__main__":
    main()
