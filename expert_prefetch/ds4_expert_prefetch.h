/* =========================================================================
 * ds4_expert_prefetch — Geometric expert prefetch for MoE streaming caches.
 * =========================================================================
 *
 * PURPOSE
 *   Reduce the number of routed experts that must stay resident in VRAM (and
 *   therefore the VRAM/RAM footprint) when running DeepSeek V4 with the ds4
 *   SSD/host expert-streaming cache, by predicting which experts will be needed
 *   soon and prefetching them.
 *
 * IDEA (borrowed from Spectral-AI, as an enhancement of streaming, NOT a router)
 *   Every routed expert e has a gate-direction vector: the e-th row of the
 *   router gate matrix `ffn_gate_inp` (shape [n_embd, n_expert], stored F16 in
 *   the GGUF). Two experts whose gate directions point the same way fire for
 *   similar hidden states, so they tend to co-activate within a token's top-k
 *   and across nearby tokens. That is exactly the "experts as points in a
 *   semantic space" geometry of Spectral-AI — but computed offline from the
 *   weights, with no RT cores and without touching the router.
 *
 *   We precompute, per expert, its top-`degree` nearest experts by cosine
 *   similarity (the "geometric neighbours"). At decode time, when the router
 *   selects an expert, we speculatively prefetch its neighbours into the
 *   streaming cache. Prefetch only ever displaces *speculative* (never-demanded)
 *   cache slots, so it can never evict an expert the model is actually using.
 *
 * WHY THIS SAVES MEMORY
 *   A better hit-rate at a fixed resident capacity means you can LOWER the
 *   resident capacity (--ssd-streaming-cache-experts) and keep the same
 *   effective hit-rate — fewer experts in VRAM = less VRAM.
 *
 * SCOPE / SAFETY
 *   This module is self-contained C99 (only <stdint.h>/<stdbool.h>/<stddef.h>
 *   in the header; <math.h>/<string.h>/<stdlib.h>/<stdio.h> in the .c). It does
 *   NOT include ds4.h and does NOT touch any existing ds4 code path. It is an
 *   opt-in component: nothing happens unless the caller builds a geometry and
 *   drives the cache. The existing static-hotlist streaming path is unchanged.
 * ========================================================================= */

#ifndef DS4_EXPERT_PREFETCH_H
#define DS4_EXPERT_PREFETCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_EXPERT_GEOMETRY_MAGIC "DS4GEO\0"   /* 8 bytes incl. NUL */
#define DS4_EXPERT_GEOMETRY_VERSION 1u

/* Per-(layer,expert) table of geometric neighbours. */
typedef struct {
    uint32_t  n_layer;
    uint32_t  n_expert;
    uint32_t  degree;     /* neighbours stored per expert */
    uint16_t *neighbor;   /* [n_layer * n_expert * degree] expert ids */
    float    *affinity;   /* [n_layer * n_expert * degree] cosine similarity */
} ds4_expert_geometry;

/* Build the neighbour table from per-layer gate-direction vectors.
 *   gate_f32[il] -> n_expert * n_embd row-major floats; row e is expert e's
 *   direction (the e-th row of ffn_gate_inp). degree must be >= 1.
 * Returns false on bad args or allocation failure. */
bool ds4_expert_geometry_build_f32(ds4_expert_geometry *geo,
                                   uint32_t n_layer, uint32_t n_expert,
                                   uint32_t n_embd, uint32_t degree,
                                   const float *const *gate_f32);

/* Same, but each layer's gate is raw IEEE-754 half precision (uint16_t),
 * exactly as ds4 stores ffn_gate_inp in memory. */
bool ds4_expert_geometry_build_f16(ds4_expert_geometry *geo,
                                   uint32_t n_layer, uint32_t n_expert,
                                   uint32_t n_embd, uint32_t degree,
                                   const uint16_t *const *gate_f16);

void ds4_expert_geometry_free(ds4_expert_geometry *geo);

/* Pointer to the `degree` neighbour ids for (layer, expert), or NULL. */
const uint16_t *ds4_expert_geometry_neighbors(const ds4_expert_geometry *geo,
                                              uint32_t layer, uint32_t expert);

/* Persist / load the binary sidecar (.ds4geo). Returns false on I/O error. */
bool ds4_expert_geometry_save(const ds4_expert_geometry *geo, const char *path);
bool ds4_expert_geometry_load(ds4_expert_geometry *geo, const char *path);

/* ----------------------------------------------------------------------- *
 * Streaming-cache policy engine (one instance per layer).
 *
 * Models a fixed-capacity resident expert set with:
 *   - a pinned set (the static hotlist) that is always resident,
 *   - an LRU-managed remainder for demand-loaded experts,
 *   - optional geometric prefetch that fills *spare/speculative* slots only.
 *
 * It is both the runtime policy object (phase 2) and the simulator core, so a
 * CPU replay measures exactly the behaviour the GPU path would get.
 * ----------------------------------------------------------------------- */
typedef struct {
    uint32_t  n_expert;
    uint32_t  cap;              /* resident capacity in experts */
    uint32_t  n_pin;            /* number of pinned (hotlist) experts */
    uint32_t  prefetch_degree;  /* neighbours to prefetch per demand (0 = off) */

    uint8_t  *resident;         /* [n_expert] */
    uint8_t  *pinned;           /* [n_expert] */
    uint64_t *last_use;         /* [n_expert] demand clock; 0 while speculative */
    uint64_t *last_touch;       /* [n_expert] any admit/use clock */
    uint32_t  count;            /* resident experts */
    uint64_t  clock;

    /* statistics (demand_* drive the VRAM argument) */
    uint64_t  demand_total;
    uint64_t  demand_hit;
    uint64_t  demand_miss;
    uint64_t  prefetch_issued;
    uint64_t  prefetch_useful;  /* prefetched then demanded before eviction */
    uint64_t  prefetch_wasted;  /* prefetched then evicted unused */
    uint64_t  bytes_streamed;   /* experts streamed in (demand + prefetch) */
} ds4_prefetch_cache;

/* pin_experts: array of n_pin expert ids to pin (e.g. the hotlist), or NULL to
 * pin nothing. cap must be >= n_pin. Returns false on bad args / alloc fail. */
bool ds4_prefetch_cache_init(ds4_prefetch_cache *c,
                             uint32_t n_expert, uint32_t cap, uint32_t n_pin,
                             uint32_t prefetch_degree,
                             const uint16_t *pin_experts);

void ds4_prefetch_cache_free(ds4_prefetch_cache *c);

/* Replay one token's routed selection at `layer`. geo may be NULL to disable
 * prefetch regardless of prefetch_degree. */
void ds4_prefetch_cache_step(ds4_prefetch_cache *c,
                             const ds4_expert_geometry *geo, uint32_t layer,
                             const int *selected, uint32_t n_selected);

/* Convenience: demand hit-rate in [0,1]. */
double ds4_prefetch_cache_hit_rate(const ds4_prefetch_cache *c);

#ifdef __cplusplus
}
#endif

#endif /* DS4_EXPERT_PREFETCH_H */
