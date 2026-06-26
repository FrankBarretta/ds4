# Geometric expert prefetch (opt-in) — save VRAM/RAM on the DeepSeek V4 streaming cache

This directory adds an **experimental, opt-in** way to reduce how many routed
experts must stay resident in VRAM (and host RAM) when running DeepSeek V4 with
ds4's SSD/host expert-streaming cache. It is the "geometric idea" from
**Spectral-AI**, applied as an *enhancement of streaming* — **not** as a router
and **not** using RT cores.

> **Nothing in the existing ds4 codebase is modified or deleted.** Everything
> here is new and self-contained. The current static-hotlist streaming path
> stays the default and untouched. You choose the new method explicitly; if you
> never opt in, ds4 behaves exactly as before.

## The idea in one paragraph

Each routed expert `e` has a *gate-direction vector*: the `e`-th row of the
router gate matrix `ffn_gate_inp` (`[n_embd, n_expert]`, F16). Experts whose gate
directions point the same way fire for similar hidden states, so they tend to
co-activate within a token's top-k and across nearby tokens. We precompute, per
expert, its nearest experts by **cosine similarity** (its "geometric
neighbours"). At decode time, when the router selects an expert, we speculatively
**prefetch its neighbours** into the streaming cache — but only into spare /
never-demanded slots, so prefetch can **never evict an expert the model is
actually using**. Better hit-rate at a fixed resident capacity ⇒ you can lower
the resident capacity (`--ssd-streaming-cache-experts`) and keep the same
effective hit-rate ⇒ **fewer experts in VRAM = less VRAM**.

Why this and not the RT-core router: in an MoE the memory is in the *experts*,
not the router; the gating is already cheap; and ds4's main targets (Apple
Silicon, AMD Strix Halo) have no RT cores. Prefetch targets the real lever —
the resident expert working set — and works on every backend.

## Files (all new)

| File | What it is |
|------|------------|
| `ds4_expert_prefetch.h/.c` | Self-contained C99 core: builds the geometric neighbour table from the gate matrix, plus the streaming-cache policy engine (pinned hotlist + LRU + safe geometric prefetch). No `ds4.h`, no GPU. |
| `expert_prefetch_sim.c` | CPU-only simulator that quantifies the VRAM saving on a synthetic-but-realistic MoE workload. **No GPU needed.** |
| `build_expert_geometry.py` | Offline generator: reads `ffn_gate_inp` from a real GGUF and writes a `.ds4geo` neighbour table. CPU only; does not modify the GGUF. |
| `Makefile` | Standalone build for the simulator (separate from the main ds4 Makefile on purpose). |

## Step 1 — validate the mechanism on CPU (no GPU, safe to run now)

```sh
cd expert_prefetch
make run
```

This builds nothing GPU-related and prints a table comparing three policies as a
function of resident cache capacity:

- `LRU`     — demand-paged LRU only (no hotlist, no prefetch)
- `HOT`     — pinned static hotlist + LRU  → **what ds4 does today**
- `GEO`     — hotlist + LRU + geometric prefetch → **the new method**

…followed by the *capacity needed to reach a target hit-rate* for HOT vs GEO,
and the resulting VRAM saved. Tune the workload to your model:

```sh
./expert_prefetch_sim --n_expert=256 --top_k=6 --degree=4 --n_pin=16 \
                      --per_expert_mib=12 --n_layer=58
```

The simulator is a model of the *mechanism*. It shows whether prefetching
geometric neighbours raises the hit-rate at a given capacity. The real numbers
come from Step 2 + Step 3 on the actual model.

## Step 2 — build the real geometry from the model (CPU only)

```sh
python3 build_expert_geometry.py /path/DeepSeek-V4-Flash-*.gguf \
        -o geometry.ds4geo --degree 4
```

This reads the real gate matrices and writes `geometry.ds4geo`
(`ds4_expert_geometry_load()` consumes it). No GPU, no model run.

## Step 3 — runtime integration into ds4 (phase 2, opt-in, NOT yet wired)

This is deliberately **not applied** to the ds4 sources yet, so your in-progress
work and the default build are untouched. When you approve it, the wiring is:

1. Add a CLI flag, **default off** (existing behaviour preserved):
   - `--expert-prefetch=geometry.ds4geo[,degree=N]` enables the new method.
   - With the flag absent, ds4 uses the current static hotlist exactly as today.
2. At load, call `ds4_expert_geometry_load()` once (tiny: `n_layer*n_expert*degree`
   uint16+float, e.g. ~1–2 MB total).
3. At the existing streaming page-in site (Metal: the "streaming prefill selected
   expert readahead" path; ROCm: the full-layer expert load), after the selected
   experts are known, also page in `ds4_expert_geometry_neighbors(layer, e)` for
   each selected `e`, bounded by `degree` and by spare cache slots — i.e. drive a
   `ds4_prefetch_cache` exactly like the simulator does.
4. Measurement: lower `--ssd-streaming-cache-experts` until the hit-rate (already
   reported by the existing expert-locality profiler) drops to your threshold,
   with and without `--expert-prefetch`. The capacity gap × per-expert bytes ×
   layers = VRAM saved.

Because the engine in `ds4_expert_prefetch.c` is the *same* object used by the
simulator, the GPU integration is a thin call site, not new policy logic.

## How "use new vs use existing" is guaranteed

- **Today:** the new method lives only in this directory and its standalone
  tools. If you do nothing, ds4 is unchanged.
- **After phase 2 (with your approval):** the `--expert-prefetch` flag selects
  it *per run*. Flag absent ⇒ the existing static-hotlist method. Flag present ⇒
  the geometric prefetch. Both coexist; neither replaces the other.

## Testing on GPU

Per your constraint: GPU testing uses **only the RTX 5060 Ti**, never the RTX
5070 Ti (training). When we wire phase 2, any CUDA run will be pinned with
`CUDA_VISIBLE_DEVICES` to the 5060 Ti's index so the 5070 Ti is never touched.
Steps 1 and 2 above need no GPU at all.
