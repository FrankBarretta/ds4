/* expert_prefetch_sim — CPU-only validation of geometric expert prefetch.
 *
 * Builds a synthetic but realistic MoE routing workload:
 *   - n_expert experts grouped into `clusters` clusters in a D-dim space;
 *     each expert's gate-direction vector = cluster centre + noise.
 *   - a token stream whose hidden state does a correlated random walk with
 *     occasional topic jumps (temporal + cluster locality, like real text).
 *   - per token, the top-k experts by gate logit (dot product) are "selected".
 *
 * It then replays the same trace through three streaming-cache policies and
 * reports the demand hit-rate as a function of resident capacity:
 *     LRU   : demand-paged LRU only            (no hotlist, no prefetch)
 *     HOT   : pinned static hotlist + LRU      (what ds4 does today)
 *     GEO   : hotlist + LRU + geometric prefetch (the new method)
 *
 * The headline result is the "equivalent capacity": the smallest resident
 * capacity each policy needs to reach a target hit-rate. If GEO reaches the
 * target with fewer resident experts than HOT, that capacity gap is the VRAM
 * saved (fewer experts kept in VRAM), which is the whole point.
 *
 * NOTE: this is a synthetic model of the mechanism. The real validation uses
 * the actual gate geometry (build_expert_geometry.py) and a real decode trace;
 * see README.md. No GPU is used or required.
 *
 * Build:  make            (in this directory)
 * Run:    ./expert_prefetch_sim [--key=value ...]
 */

#include "ds4_expert_prefetch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- deterministic RNG (fixed seed => reproducible) --------------------- */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t xorshift64(void) {
    uint64_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng = x;
    return x;
}
static double rnd_unit(void) {              /* [0,1) */
    return (double)(xorshift64() >> 11) * (1.0 / 9007199254740992.0);
}
static double rnd_normal(void) {
    double u1 = rnd_unit(), u2 = rnd_unit();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ---- config ------------------------------------------------------------- */
typedef struct {
    uint32_t n_expert;
    uint32_t dim;          /* gate-direction dimensionality (stand-in n_embd) */
    uint32_t clusters;
    uint32_t top_k;        /* experts selected per token */
    uint32_t degree;       /* geometric neighbours per expert */
    uint32_t n_pin;        /* hotlist size */
    uint32_t trace_len;
    uint32_t warmup;
    double   noise;        /* expert spread within a cluster */
    double   walk;         /* per-token drift of the context vector */
    double   jump_prob;    /* topic-switch probability per token */
    double   per_expert_mib;
    uint32_t n_layer;      /* layers, for the total-VRAM extrapolation */
} sim_cfg;

static void cfg_defaults(sim_cfg *c) {
    c->n_expert       = 256;
    c->dim            = 64;
    c->clusters       = 12;
    c->top_k          = 6;
    c->degree         = 4;
    c->n_pin          = 16;
    c->trace_len      = 40000;
    c->warmup         = 4000;
    c->noise          = 0.55;
    c->walk           = 0.18;
    c->jump_prob      = 0.012;
    c->per_expert_mib = 12.0;  /* ~DeepSeek V4 Flash IQ2 routed expert */
    c->n_layer        = 58;
}

static void cfg_apply(sim_cfg *c, const char *k, const char *v) {
    if      (!strcmp(k, "n_expert"))       c->n_expert = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "dim"))            c->dim = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "clusters"))       c->clusters = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "top_k"))          c->top_k = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "degree"))         c->degree = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "n_pin"))          c->n_pin = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "trace_len"))      c->trace_len = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "warmup"))         c->warmup = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "noise"))          c->noise = strtod(v, 0);
    else if (!strcmp(k, "walk"))           c->walk = strtod(v, 0);
    else if (!strcmp(k, "jump_prob"))      c->jump_prob = strtod(v, 0);
    else if (!strcmp(k, "per_expert_mib")) c->per_expert_mib = strtod(v, 0);
    else if (!strcmp(k, "n_layer"))        c->n_layer = (uint32_t)strtoul(v, 0, 10);
    else if (!strcmp(k, "seed"))           g_rng = strtoull(v, 0, 10) | 1ull;
    else fprintf(stderr, "warning: unknown option '%s'\n", k);
}

/* ---- trace + scoring ---------------------------------------------------- */
static void topk_by_score(const double *score, uint32_t n, uint32_t k,
                          int *out) {
    /* simple partial selection; k is tiny */
    char *used = (char *)calloc(n, 1);
    for (uint32_t r = 0; r < k; r++) {
        int best = -1;
        double bv = -1e300;
        for (uint32_t e = 0; e < n; e++) {
            if (used[e]) continue;
            if (score[e] > bv) { bv = score[e]; best = (int)e; }
        }
        out[r] = best;
        if (best >= 0) used[best] = 1;
    }
    free(used);
}

int main(int argc, char **argv) {
    sim_cfg cfg;
    cfg_defaults(&cfg);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a += 2;
        const char *eq = strchr(a, '=');
        if (!eq) { fprintf(stderr, "ignoring '%s' (use --key=value)\n", argv[i]); continue; }
        char key[64];
        size_t kl = (size_t)(eq - a);
        if (kl >= sizeof(key)) kl = sizeof(key) - 1;
        memcpy(key, a, kl);
        key[kl] = '\0';
        cfg_apply(&cfg, key, eq + 1);
    }

    const uint32_t N = cfg.n_expert, D = cfg.dim, K = cfg.top_k;
    if (K > N) { fprintf(stderr, "top_k > n_expert\n"); return 2; }
    if (cfg.trace_len <= cfg.warmup) { fprintf(stderr, "trace_len <= warmup\n"); return 2; }

    printf("expert_prefetch_sim — geometric expert prefetch (synthetic workload)\n");
    printf("  experts=%u dim=%u clusters=%u top_k=%u  degree=%u hotlist=%u\n",
           N, D, cfg.clusters, K, cfg.degree, cfg.n_pin);
    printf("  trace=%u (warmup=%u) noise=%.2f walk=%.2f jump=%.3f\n\n",
           cfg.trace_len, cfg.warmup, cfg.noise, cfg.walk, cfg.jump_prob);

    /* cluster centres + expert vectors */
    double *centre = (double *)malloc((size_t)cfg.clusters * D * sizeof(double));
    float  *V      = (float  *)malloc((size_t)N * D * sizeof(float));
    for (uint32_t g = 0; g < cfg.clusters; g++)
        for (uint32_t d = 0; d < D; d++)
            centre[(size_t)g * D + d] = rnd_normal();
    for (uint32_t e = 0; e < N; e++) {
        uint32_t g = e % cfg.clusters;
        for (uint32_t d = 0; d < D; d++)
            V[(size_t)e * D + d] =
                (float)(centre[(size_t)g * D + d] + cfg.noise * rnd_normal());
    }

    /* geometry from the (single-layer) expert vectors */
    const float *gate_ptr = V;
    ds4_expert_geometry geo;
    if (!ds4_expert_geometry_build_f32(&geo, 1, N, D, cfg.degree, &gate_ptr)) {
        fprintf(stderr, "geometry build failed\n");
        return 1;
    }

    /* generate the trace */
    int *trace = (int *)malloc((size_t)cfg.trace_len * K * sizeof(int));
    uint64_t *freq = (uint64_t *)calloc(N, sizeof(uint64_t));
    double *x = (double *)malloc((size_t)D * sizeof(double));
    double *score = (double *)malloc((size_t)N * sizeof(double));
    for (uint32_t d = 0; d < D; d++) x[d] = rnd_normal();

    for (uint32_t t = 0; t < cfg.trace_len; t++) {
        if (rnd_unit() < cfg.jump_prob) {
            for (uint32_t d = 0; d < D; d++) x[d] = rnd_normal();
        } else {
            for (uint32_t d = 0; d < D; d++) x[d] += cfg.walk * rnd_normal();
        }
        for (uint32_t e = 0; e < N; e++) {
            const float *ve = V + (size_t)e * D;
            double s = 0.0;
            for (uint32_t d = 0; d < D; d++) s += x[d] * ve[d];
            score[e] = s;
        }
        int *sel = trace + (size_t)t * K;
        topk_by_score(score, N, K, sel);
        if (t >= cfg.warmup)
            for (uint32_t r = 0; r < K; r++)
                if (sel[r] >= 0) freq[sel[r]]++;
    }

    /* hotlist = most frequent experts over the measured window */
    uint16_t *order = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    for (uint32_t e = 0; e < N; e++) order[e] = (uint16_t)e;
    for (uint32_t a = 0; a < N; a++)         /* selection sort by freq desc */
        for (uint32_t b = a + 1; b < N; b++)
            if (freq[order[b]] > freq[order[a]]) {
                uint16_t tmp = order[a]; order[a] = order[b]; order[b] = tmp;
            }
    uint32_t pin = cfg.n_pin < N ? cfg.n_pin : N;

    /* ---- sweep capacities -------------------------------------------- */
    static const double TARGETS[] = { 0.90, 0.95, 0.99 };
    const uint32_t n_targets = (uint32_t)(sizeof(TARGETS) / sizeof(TARGETS[0]));
    uint32_t cap_lru[3], cap_hot[3], cap_geo[3];
    for (uint32_t i = 0; i < n_targets; i++) { cap_lru[i] = cap_hot[i] = cap_geo[i] = 0; }

    printf("  cap   LRU      HOT(ds4)  GEO(new)   GEO useful/issued\n");
    printf("  ----  -------  --------  --------   -----------------\n");

    for (uint32_t cap = (K > pin ? K : pin); cap <= N; cap += (N > 64 ? 8 : 4)) {
        double hr[3] = { 0, 0, 0 };
        uint64_t useful = 0, issued = 0;

        for (int mode = 0; mode < 3; mode++) {
            ds4_prefetch_cache c;
            uint32_t use_pin = (mode == 0) ? 0 : pin;
            uint32_t use_deg = (mode == 2) ? cfg.degree : 0;
            const uint16_t *pinp = (mode == 0) ? NULL : order;
            if (cap < use_pin) { hr[mode] = -1.0; continue; }
            if (!ds4_prefetch_cache_init(&c, N, cap, use_pin, use_deg, pinp)) {
                hr[mode] = -1.0; continue;
            }
            for (uint32_t t = cfg.warmup; t < cfg.trace_len; t++)
                ds4_prefetch_cache_step(&c, (mode == 2) ? &geo : NULL, 0,
                                        trace + (size_t)t * K, K);
            hr[mode] = ds4_prefetch_cache_hit_rate(&c);
            if (mode == 2) { useful = c.prefetch_useful; issued = c.prefetch_issued; }
            ds4_prefetch_cache_free(&c);
        }

        for (uint32_t i = 0; i < n_targets; i++) {
            if (hr[0] >= 0 && cap_lru[i] == 0 && hr[0] >= TARGETS[i]) cap_lru[i] = cap;
            if (hr[1] >= 0 && cap_hot[i] == 0 && hr[1] >= TARGETS[i]) cap_hot[i] = cap;
            if (hr[2] >= 0 && cap_geo[i] == 0 && hr[2] >= TARGETS[i]) cap_geo[i] = cap;
        }

        double frac = issued ? (double)useful / (double)issued : 0.0;
        printf("  %4u  %6.2f%%  %6.2f%%  %6.2f%%   %5.1f%% (%llu/%llu)\n",
               cap, 100.0 * hr[0], 100.0 * hr[1], 100.0 * hr[2],
               100.0 * frac,
               (unsigned long long)useful, (unsigned long long)issued);
    }

    /* ---- VRAM-saving summary ----------------------------------------- */
    printf("\n  Capacity needed to reach a target demand hit-rate (per layer):\n");
    printf("  target  HOT(ds4)  GEO(new)  saved/layer   saved total (%u layers)\n",
           cfg.n_layer);
    printf("  ------  --------  --------  -----------   ------------------------\n");
    for (uint32_t i = 0; i < n_targets; i++) {
        uint32_t h = cap_hot[i], g = cap_geo[i];
        if (h == 0 || g == 0) {
            printf("  %4.0f%%   %s        %s\n", 100.0 * TARGETS[i],
                   h ? "  -" : "n/a", g ? "  -" : "n/a");
            continue;
        }
        long dl = (long)h - (long)g;
        double per_layer_mib = (double)dl * cfg.per_expert_mib;
        double total_gib = per_layer_mib * (double)cfg.n_layer / 1024.0;
        printf("  %4.0f%%   %6u    %6u    %+6ld exp     %+7.2f GiB\n",
               100.0 * TARGETS[i], h, g, dl, total_gib);
    }
    printf("\n  (per-expert size assumed %.1f MiB; pass --per_expert_mib / --n_layer\n"
           "   to match your real model. Positive 'saved' = less VRAM resident.)\n",
           cfg.per_expert_mib);

    free(centre); free(V); free(trace); free(freq);
    free(x); free(score); free(order);
    ds4_expert_geometry_free(&geo);
    return 0;
}
