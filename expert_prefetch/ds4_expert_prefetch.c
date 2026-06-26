/* ds4_expert_prefetch — see ds4_expert_prefetch.h for the design rationale.
 *
 * Self-contained C99. No dependency on ds4.h or any GPU backend. */

#include "ds4_expert_prefetch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- IEEE-754 half -> float (no hardware/_Float16 assumptions) ---------- */
static float ds4ep_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       /* +/- zero */
        } else {
            /* subnormal half -> normalized float */
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / nan */
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* ---- geometry construction --------------------------------------------- */

/* Insert (id, sim) into a descending-by-sim top-`degree` list kept in
 * nb_id[0..degree-1] / nb_sim[0..degree-1]; *filled tracks how many are valid. */
static void ds4ep_topk_insert(uint16_t *nb_id, float *nb_sim, uint32_t degree,
                              uint32_t *filled, uint16_t id, float sim) {
    if (*filled < degree) {
        uint32_t i = (*filled)++;
        while (i > 0 && nb_sim[i - 1] < sim) {
            nb_sim[i] = nb_sim[i - 1];
            nb_id[i]  = nb_id[i - 1];
            i--;
        }
        nb_sim[i] = sim;
        nb_id[i]  = id;
        return;
    }
    if (sim <= nb_sim[degree - 1]) return;     /* not good enough */
    uint32_t i = degree - 1;
    while (i > 0 && nb_sim[i - 1] < sim) {
        nb_sim[i] = nb_sim[i - 1];
        nb_id[i]  = nb_id[i - 1];
        i--;
    }
    nb_sim[i] = sim;
    nb_id[i]  = id;
}

/* Fill one layer's neighbour rows from L2-normalized expert vectors `norm`
 * (n_expert * n_embd, row-major). */
static void ds4ep_build_layer(const float *norm, uint32_t n_expert,
                              uint32_t n_embd, uint32_t degree,
                              uint16_t *neighbor_out, float *affinity_out) {
    for (uint32_t i = 0; i < n_expert; i++) {
        const float *vi = norm + (size_t)i * n_embd;
        uint16_t *nb_id  = neighbor_out + (size_t)i * degree;
        float    *nb_sim = affinity_out + (size_t)i * degree;
        uint32_t  filled = 0;

        for (uint32_t j = 0; j < n_expert; j++) {
            if (j == i) continue;
            const float *vj = norm + (size_t)j * n_embd;
            float dot = 0.0f;
            for (uint32_t k = 0; k < n_embd; k++) dot += vi[k] * vj[k];
            ds4ep_topk_insert(nb_id, nb_sim, degree, &filled, (uint16_t)j, dot);
        }
        /* pad any unfilled slots (n_expert <= degree) with self => no-op
         * prefetch target (already resident on demand). */
        for (uint32_t r = filled; r < degree; r++) {
            nb_id[r]  = (uint16_t)i;
            nb_sim[r] = -2.0f;
        }
    }
}

static bool ds4ep_alloc_geo(ds4_expert_geometry *geo, uint32_t n_layer,
                            uint32_t n_expert, uint32_t degree) {
    memset(geo, 0, sizeof(*geo));
    if (n_layer == 0 || n_expert == 0 || degree == 0) return false;
    if (n_expert > 0xFFFFu) return false;       /* ids must fit uint16_t */

    const size_t n = (size_t)n_layer * n_expert * degree;
    geo->neighbor = (uint16_t *)malloc(n * sizeof(uint16_t));
    geo->affinity = (float *)malloc(n * sizeof(float));
    if (!geo->neighbor || !geo->affinity) {
        free(geo->neighbor);
        free(geo->affinity);
        memset(geo, 0, sizeof(*geo));
        return false;
    }
    geo->n_layer  = n_layer;
    geo->n_expert = n_expert;
    geo->degree   = degree;
    return true;
}

bool ds4_expert_geometry_build_f32(ds4_expert_geometry *geo,
                                   uint32_t n_layer, uint32_t n_expert,
                                   uint32_t n_embd, uint32_t degree,
                                   const float *const *gate_f32) {
    if (!geo || !gate_f32 || n_embd == 0) return false;
    if (!ds4ep_alloc_geo(geo, n_layer, n_expert, degree)) return false;

    float *norm = (float *)malloc((size_t)n_expert * n_embd * sizeof(float));
    if (!norm) { ds4_expert_geometry_free(geo); return false; }

    for (uint32_t l = 0; l < n_layer; l++) {
        const float *g = gate_f32[l];
        for (uint32_t e = 0; e < n_expert; e++) {
            const float *src = g + (size_t)e * n_embd;
            float *dst = norm + (size_t)e * n_embd;
            double ss = 0.0;
            for (uint32_t k = 0; k < n_embd; k++) ss += (double)src[k] * src[k];
            float inv = ss > 0.0 ? (float)(1.0 / sqrt(ss)) : 0.0f;
            for (uint32_t k = 0; k < n_embd; k++) dst[k] = src[k] * inv;
        }
        ds4ep_build_layer(norm, n_expert, n_embd, degree,
                          geo->neighbor + (size_t)l * n_expert * degree,
                          geo->affinity + (size_t)l * n_expert * degree);
    }

    free(norm);
    return true;
}

bool ds4_expert_geometry_build_f16(ds4_expert_geometry *geo,
                                   uint32_t n_layer, uint32_t n_expert,
                                   uint32_t n_embd, uint32_t degree,
                                   const uint16_t *const *gate_f16) {
    if (!geo || !gate_f16 || n_embd == 0) return false;
    if (!ds4ep_alloc_geo(geo, n_layer, n_expert, degree)) return false;

    float *norm = (float *)malloc((size_t)n_expert * n_embd * sizeof(float));
    if (!norm) { ds4_expert_geometry_free(geo); return false; }

    for (uint32_t l = 0; l < n_layer; l++) {
        const uint16_t *g = gate_f16[l];
        for (uint32_t e = 0; e < n_expert; e++) {
            const uint16_t *src = g + (size_t)e * n_embd;
            float *dst = norm + (size_t)e * n_embd;
            double ss = 0.0;
            for (uint32_t k = 0; k < n_embd; k++) {
                float v = ds4ep_half_to_float(src[k]);
                dst[k] = v;
                ss += (double)v * v;
            }
            float inv = ss > 0.0 ? (float)(1.0 / sqrt(ss)) : 0.0f;
            for (uint32_t k = 0; k < n_embd; k++) dst[k] *= inv;
        }
        ds4ep_build_layer(norm, n_expert, n_embd, degree,
                          geo->neighbor + (size_t)l * n_expert * degree,
                          geo->affinity + (size_t)l * n_expert * degree);
    }

    free(norm);
    return true;
}

void ds4_expert_geometry_free(ds4_expert_geometry *geo) {
    if (!geo) return;
    free(geo->neighbor);
    free(geo->affinity);
    memset(geo, 0, sizeof(*geo));
}

const uint16_t *ds4_expert_geometry_neighbors(const ds4_expert_geometry *geo,
                                              uint32_t layer, uint32_t expert) {
    if (!geo || !geo->neighbor) return NULL;
    if (layer >= geo->n_layer || expert >= geo->n_expert) return NULL;
    return geo->neighbor +
           ((size_t)layer * geo->n_expert + expert) * geo->degree;
}

/* ---- binary sidecar ----------------------------------------------------- */

bool ds4_expert_geometry_save(const ds4_expert_geometry *geo, const char *path) {
    if (!geo || !geo->neighbor || !path) return false;
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;

    const uint32_t hdr[4] = { DS4_EXPERT_GEOMETRY_VERSION, geo->n_layer,
                              geo->n_expert, geo->degree };
    const size_t n = (size_t)geo->n_layer * geo->n_expert * geo->degree;
    bool ok =
        fwrite(DS4_EXPERT_GEOMETRY_MAGIC, 1, 8, fp) == 8 &&
        fwrite(hdr, sizeof(uint32_t), 4, fp) == 4 &&
        fwrite(geo->neighbor, sizeof(uint16_t), n, fp) == n &&
        fwrite(geo->affinity, sizeof(float), n, fp) == n;

    if (fclose(fp) != 0) ok = false;
    return ok;
}

bool ds4_expert_geometry_load(ds4_expert_geometry *geo, const char *path) {
    if (!geo || !path) return false;
    memset(geo, 0, sizeof(*geo));
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    char magic[8];
    uint32_t hdr[4];
    if (fread(magic, 1, 8, fp) != 8 ||
        memcmp(magic, DS4_EXPERT_GEOMETRY_MAGIC, 8) != 0 ||
        fread(hdr, sizeof(uint32_t), 4, fp) != 4 ||
        hdr[0] != DS4_EXPERT_GEOMETRY_VERSION) {
        fclose(fp);
        return false;
    }
    if (!ds4ep_alloc_geo(geo, hdr[1], hdr[2], hdr[3])) { fclose(fp); return false; }

    const size_t n = (size_t)geo->n_layer * geo->n_expert * geo->degree;
    bool ok = fread(geo->neighbor, sizeof(uint16_t), n, fp) == n &&
              fread(geo->affinity, sizeof(float), n, fp) == n;
    fclose(fp);
    if (!ok) ds4_expert_geometry_free(geo);
    return ok;
}

/* ---- streaming-cache policy engine -------------------------------------- */

bool ds4_prefetch_cache_init(ds4_prefetch_cache *c,
                             uint32_t n_expert, uint32_t cap, uint32_t n_pin,
                             uint32_t prefetch_degree,
                             const uint16_t *pin_experts) {
    if (!c || n_expert == 0 || cap == 0 || cap < n_pin || cap > n_expert)
        return false;
    if (n_pin > 0 && !pin_experts) return false;

    memset(c, 0, sizeof(*c));
    c->n_expert        = n_expert;
    c->cap             = cap;
    c->prefetch_degree = prefetch_degree;
    c->resident   = (uint8_t  *)calloc(n_expert, sizeof(uint8_t));
    c->pinned     = (uint8_t  *)calloc(n_expert, sizeof(uint8_t));
    c->last_use   = (uint64_t *)calloc(n_expert, sizeof(uint64_t));
    c->last_touch = (uint64_t *)calloc(n_expert, sizeof(uint64_t));
    if (!c->resident || !c->pinned || !c->last_use || !c->last_touch) {
        ds4_prefetch_cache_free(c);
        return false;
    }

    for (uint32_t i = 0; i < n_pin; i++) {
        uint16_t e = pin_experts[i];
        if (e >= n_expert || c->pinned[e]) continue; /* skip dup / oob */
        c->pinned[e]   = 1;
        c->resident[e] = 1;
        c->count++;
        c->n_pin++;
    }
    return true;
}

void ds4_prefetch_cache_free(ds4_prefetch_cache *c) {
    if (!c) return;
    free(c->resident);
    free(c->pinned);
    free(c->last_use);
    free(c->last_touch);
    memset(c, 0, sizeof(*c));
}

/* Evict one non-pinned resident. If `speculative_only`, only an outstanding
 * prefetch (resident, non-pinned, never demanded) may be evicted; returns true
 * if a victim was freed. Otherwise evicts the best LRU victim, preferring
 * speculative entries, then least-recently-demanded. */
static bool ds4ep_evict(ds4_prefetch_cache *c, bool speculative_only) {
    uint32_t victim = UINT32_MAX;
    uint64_t best_key = 0;
    bool victim_spec = false;

    for (uint32_t e = 0; e < c->n_expert; e++) {
        if (!c->resident[e] || c->pinned[e]) continue;
        bool spec = (c->last_use[e] == 0);
        if (speculative_only && !spec) continue;

        /* Rank: speculative entries first (evict before demanded), then by the
         * relevant recency clock ascending (oldest = best victim). */
        uint64_t key = spec ? c->last_touch[e] : c->last_use[e];
        if (victim == UINT32_MAX ||
            (spec && !victim_spec) ||
            (spec == victim_spec && key < best_key)) {
            victim = e;
            best_key = key;
            victim_spec = spec;
        }
    }
    if (victim == UINT32_MAX) return false;

    if (victim_spec) c->prefetch_wasted++;       /* prefetched, never used */
    c->resident[victim]   = 0;
    c->last_use[victim]   = 0;
    c->last_touch[victim] = 0;
    c->count--;
    return true;
}

void ds4_prefetch_cache_step(ds4_prefetch_cache *c,
                             const ds4_expert_geometry *geo, uint32_t layer,
                             const int *selected, uint32_t n_selected) {
    if (!c || !selected) return;
    c->clock++;

    /* Phase A: demands. A demand may evict anything non-pinned. */
    for (uint32_t i = 0; i < n_selected; i++) {
        int s = selected[i];
        if (s < 0 || (uint32_t)s >= c->n_expert) continue;
        uint32_t e = (uint32_t)s;

        c->demand_total++;
        if (c->resident[e]) {
            c->demand_hit++;
            if (!c->pinned[e] && c->last_use[e] == 0) c->prefetch_useful++;
        } else {
            c->demand_miss++;
            if (c->count >= c->cap) ds4ep_evict(c, false);
            c->resident[e] = 1;
            c->count++;
            c->bytes_streamed++;
        }
        c->last_use[e]   = c->clock;   /* now a demanded (protected) entry */
        c->last_touch[e] = c->clock;
    }

    /* Phase B: geometric prefetch into spare/speculative slots only. */
    uint32_t deg = c->prefetch_degree;
    if (!geo || deg == 0) return;
    if (deg > geo->degree) deg = geo->degree;

    for (uint32_t i = 0; i < n_selected; i++) {
        int s = selected[i];
        if (s < 0 || (uint32_t)s >= c->n_expert) continue;
        const uint16_t *nb = ds4_expert_geometry_neighbors(geo, layer,
                                                           (uint32_t)s);
        if (!nb) continue;

        for (uint32_t r = 0; r < deg; r++) {
            uint32_t e = nb[r];
            if (e >= c->n_expert || c->resident[e]) continue;
            if (c->count >= c->cap && !ds4ep_evict(c, true)) {
                /* cache full of pinned/demanded experts: no speculative room */
                break;
            }
            c->resident[e]   = 1;
            c->last_touch[e] = c->clock;
            c->last_use[e]   = 0;        /* speculative until demanded */
            c->count++;
            c->prefetch_issued++;
            c->bytes_streamed++;
        }
    }
}

double ds4_prefetch_cache_hit_rate(const ds4_prefetch_cache *c) {
    if (!c || c->demand_total == 0) return 0.0;
    return (double)c->demand_hit / (double)c->demand_total;
}
