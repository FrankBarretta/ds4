#!/usr/bin/env python3
"""Compute the real on-disk bytes of one routed expert in a DeepSeek V4 GGUF,
by reading the blk.0 ffn_{gate,up,down}_exps tensor infos. CPU only, reads only
the header."""
import struct, sys

GGUF = 0x46554747
SIMPLE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
# ggml type -> (bytes per block, elements per block)
BLK = {0:(4,1), 1:(2,1), 8:(34,32), 10:(84,256), 16:(66,256), 11:(110,256)}
TNAME = {0:"F32",1:"F16",8:"Q8_0",10:"Q2_K",16:"IQ2_XXS",11:"Q3_K"}

def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
def rstr(f):
    (n,) = rd(f, "<Q"); return f.read(n).decode("utf-8", "replace")
def rval(f, t):
    if t in SIMPLE: f.read(SIMPLE[t]); return
    if t == 8: rstr(f); return
    if t == 9:
        (et,) = rd(f, "<I"); (c,) = rd(f, "<Q")
        for _ in range(c): rval(f, et)
        return
    raise ValueError(t)

path = sys.argv[1]
f = open(path, "rb")
magic, ver = rd(f, "<II"); assert magic == GGUF, "not gguf"
nt, nkv = rd(f, "<QQ")
align = 32
for _ in range(nkv):
    k = rstr(f); (t,) = rd(f, "<I")
    if k == "general.alignment" and t == 4:
        (align,) = rd(f, "<I")
    else:
        rval(f, t)

want = {"blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight"}
found = {}
for _ in range(nt):
    name = rstr(f); (nd,) = rd(f, "<I")
    dims = [rd(f, "<Q")[0] for _ in range(nd)]
    (tt,) = rd(f, "<I"); (off,) = rd(f, "<Q")
    if name in want:
        found[name] = (dims, tt)

per_expert = 0
n_expert = None
for name in ("blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight"):
    dims, tt = found[name]
    ne = int(dims[-1])               # last dim = expert count
    n_expert = ne
    elems = 1
    for d in dims: elems *= int(d)
    bb, epb = BLK[tt]
    total = elems // epb * bb
    pe = total // ne
    per_expert += pe
    print(f"{name}: dims={dims} type={TNAME.get(tt,tt)} per_expert={pe/1024/1024:.3f} MiB")

print(f"n_expert per layer = {n_expert}")
print(f"PER_EXPERT_TOTAL = {per_expert/1024/1024:.3f} MiB (gate+up+down)")
