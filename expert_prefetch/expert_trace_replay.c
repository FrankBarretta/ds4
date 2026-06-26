/* expert_trace_replay — measure geometric expert prefetch on a REAL trace.
 *
 * Unlike expert_prefetch_sim.c (which invents a synthetic workload), this tool
 * replays an actual per-token expert-selection trace captured from a real ds4
 * decode run, using the REAL gate geometry (geometry.ds4geo). It models ds4's
 * streaming cache as a single GLOBAL pool keyed by (layer, expert) with a total
 * capacity, matching how ds4 budgets a fixed number of resident experts across
 * all layers, and reports the demand hit-rate vs capacity for:
 *     LRU   : demand-paged LRU only
 *     HOT   : pinned static hotlist + LRU      (≈ what ds4 does today)
 *     GEO   : hotlist + LRU + geometric prefetch (the new method)
 * plus the capacity (= resident experts = VRAM/RAM) each needs for a target
 * hit-rate, and the resulting saving.
 *
 * Trace file format (text), one routed selection per line, in decode order:
 *     # ds4 expert trace v1 n_layer=43 n_expert=256
 *     <layer> <e0> <e1> ... <e_{k-1}>
 * Comment/blank lines are ignored. A (layer,expert) is mapped to the global id
 * layer*n_expert + expert.
 *
 * Build:  make trace-replay   (in this directory)
 * Run:    ./expert_trace_replay --geo=geometry.ds4geo --trace=trace.txt
 *         ./expert_trace_replay --geo=geometry.ds4geo --synth=2000   (smoke test)
 *
 * No GPU. The --synth mode builds a trace FROM the geometry, so its numbers are
 * circular and only validate that the tool runs; report real numbers only.
 */

#include "ds4_expert_prefetch.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXK 16

static uint64_t g_rng = 0xDEADBEEF12345678ull;
static uint64_t xs64(void){uint64_t x=g_rng;x^=x<<13;x^=x>>7;x^=x<<17;g_rng=x;return x;}
static uint32_t rnd_below(uint32_t n){return n?(uint32_t)(xs64()%n):0;}

typedef struct {
    int      *ids;    /* flattened global expert ids */
    int      *nsel;   /* per-step selection count */
    size_t    nsteps;
    size_t    cap_steps;
    size_t    cap_ids;
    size_t    ids_len;
} trace_t;

static void trace_push(trace_t *t, const int *ids, int n) {
    if (t->nsteps == t->cap_steps) {
        t->cap_steps = t->cap_steps ? t->cap_steps * 2 : 4096;
        t->nsel = realloc(t->nsel, t->cap_steps * sizeof(int));
    }
    if (t->ids_len + (size_t)n > t->cap_ids) {
        while (t->ids_len + (size_t)n > t->cap_ids)
            t->cap_ids = t->cap_ids ? t->cap_ids * 2 : 65536;
        t->ids = realloc(t->ids, t->cap_ids * sizeof(int));
    }
    memcpy(t->ids + t->ids_len, ids, (size_t)n * sizeof(int));
    t->ids_len += (size_t)n;
    t->nsel[t->nsteps++] = n;
}

/* Flatten per-layer geometry into a single global (1-layer) neighbour table
 * whose ids live in [0, n_layer*n_expert). */
static int build_global_geo(const ds4_expert_geometry *per_layer,
                            ds4_expert_geometry *global) {
    const uint32_t nl = per_layer->n_layer, ne = per_layer->n_expert,
                   deg = per_layer->degree;
    const size_t total = (size_t)nl * ne;
    memset(global, 0, sizeof(*global));
    global->n_layer = 1;
    global->n_expert = (uint32_t)total;
    global->degree = deg;
    global->neighbor = malloc(total * deg * sizeof(uint16_t));
    global->affinity = malloc(total * deg * sizeof(float));
    if (!global->neighbor || !global->affinity) return 0;
    if (total > 0xFFFFu) {
        /* ids exceed uint16_t range used by the neighbour table; this tool
         * needs a wider id type. 43*256=11008 < 65536, so we are fine here. */
        fprintf(stderr, "warning: global id space %zu exceeds uint16_t\n", total);
    }
    for (uint32_t l = 0; l < nl; l++) {
        for (uint32_t e = 0; e < ne; e++) {
            const uint16_t *src =
                per_layer->neighbor + ((size_t)l * ne + e) * deg;
            const float *srca =
                per_layer->affinity + ((size_t)l * ne + e) * deg;
            size_t base = ((size_t)l * ne + e) * deg;
            for (uint32_t r = 0; r < deg; r++) {
                global->neighbor[base + r] = (uint16_t)(l * ne + src[r]);
                global->affinity[base + r] = srca[r];
            }
        }
    }
    return 1;
}

static int read_trace(const char *path, uint32_t ne, trace_t *t) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open trace %s\n", path); return 0; }
    char line[4096];
    int ids[MAXK];
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        long vals[1 + MAXK];
        int nv = 0;
        char *end;
        while (nv < 1 + MAXK) {
            long v = strtol(p, &end, 10);
            if (end == p) break;
            vals[nv++] = v;
            p = end;
        }
        if (nv < 2) continue;                 /* need layer + >=1 expert */
        long layer = vals[0];
        int n = 0;
        for (int i = 1; i < nv && n < MAXK; i++) {
            long e = vals[i];
            if (e < 0 || e >= (long)ne) continue;
            ids[n++] = (int)(layer * (long)ne + e);
        }
        if (n > 0) trace_push(t, ids, n);
    }
    fclose(fp);
    return 1;
}

/* Build a circular-but-runnable synthetic trace from the real geometry. */
static void synth_trace(const ds4_expert_geometry *g, uint32_t tokens,
                        uint32_t top_k, trace_t *t) {
    const uint32_t nl = g->n_layer, ne = g->n_expert;
    uint32_t *seed = calloc(nl, sizeof(uint32_t));
    for (uint32_t l = 0; l < nl; l++) seed[l] = rnd_below(ne);
    int ids[MAXK];
    for (uint32_t tok = 0; tok < tokens; tok++) {
        for (uint32_t l = 0; l < nl; l++) {
            if (rnd_below(100) < 3) seed[l] = rnd_below(ne);  /* topic jump */
            else {
                const uint16_t *nb = ds4_expert_geometry_neighbors(g, l, seed[l]);
                if (nb) seed[l] = nb[rnd_below(g->degree)];   /* drift to a neighbour */
            }
            const uint16_t *nb = ds4_expert_geometry_neighbors(g, l, seed[l]);
            int n = 0;
            ids[n++] = (int)(l * ne + seed[l]);
            for (uint32_t r = 0; r < g->degree && (uint32_t)n < top_k; r++)
                ids[n++] = (int)(l * ne + nb[r]);
            trace_push(t, ids, n);
        }
    }
    free(seed);
}

typedef struct { uint32_t id; uint64_t f; } freq_t;
static int freq_cmp(const void *a, const void *b) {
    const freq_t *x = a, *y = b;
    if (x->f != y->f) return x->f < y->f ? 1 : -1;
    return x->id < y->id ? -1 : 1;
}

int main(int argc, char **argv) {
    const char *geo_path = "geometry.ds4geo";
    const char *trace_path = NULL;
    uint32_t synth = 0, top_k = 6, n_pin = 256, degree = 0;
    uint32_t cap_min = 0, cap_max = 0, cap_step = 0;
    double per_expert_mib = 12.0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "--geo=", 6)) geo_path = a + 6;
        else if (!strncmp(a, "--trace=", 8)) trace_path = a + 8;
        else if (!strncmp(a, "--synth=", 8)) synth = (uint32_t)strtoul(a + 8, 0, 10);
        else if (!strncmp(a, "--top_k=", 8)) top_k = (uint32_t)strtoul(a + 8, 0, 10);
        else if (!strncmp(a, "--n_pin=", 8)) n_pin = (uint32_t)strtoul(a + 8, 0, 10);
        else if (!strncmp(a, "--degree=", 9)) degree = (uint32_t)strtoul(a + 9, 0, 10);
        else if (!strncmp(a, "--per_expert_mib=", 17)) per_expert_mib = strtod(a + 17, 0);
        else if (!strncmp(a, "--cap_min=", 10)) cap_min = (uint32_t)strtoul(a + 10, 0, 10);
        else if (!strncmp(a, "--cap_max=", 10)) cap_max = (uint32_t)strtoul(a + 10, 0, 10);
        else if (!strncmp(a, "--cap_step=", 11)) cap_step = (uint32_t)strtoul(a + 11, 0, 10);
        else fprintf(stderr, "ignoring '%s'\n", a);
    }
    if (top_k > MAXK) top_k = MAXK;

    ds4_expert_geometry geo;
    if (!ds4_expert_geometry_load(&geo, geo_path)) {
        fprintf(stderr, "failed to load geometry %s\n", geo_path);
        return 1;
    }
    if (degree == 0 || degree > geo.degree) degree = geo.degree;
    printf("geometry: %u layers x %u experts x %u neighbours (using degree=%u)\n",
           geo.n_layer, geo.n_expert, geo.degree, degree);

    ds4_expert_geometry gg;
    if (!build_global_geo(&geo, &gg)) { fprintf(stderr, "oom\n"); return 1; }
    const uint32_t NTOT = gg.n_expert;   /* global (layer,expert) id space */

    trace_t tr; memset(&tr, 0, sizeof(tr));
    if (trace_path) {
        if (!read_trace(trace_path, geo.n_expert, &tr)) return 1;
        printf("trace: %zu routed selections from %s\n", tr.nsteps, trace_path);
    } else if (synth) {
        synth_trace(&geo, synth, top_k, &tr);
        printf("trace: %zu routed selections (SYNTHETIC from geometry — "
               "numbers are circular, tool smoke-test only)\n", tr.nsteps);
    } else {
        fprintf(stderr, "need --trace=FILE or --synth=N\n");
        return 2;
    }
    if (tr.nsteps == 0) { fprintf(stderr, "empty trace\n"); return 1; }

    /* global frequency -> hotlist */
    freq_t *fr = malloc((size_t)NTOT * sizeof(freq_t));
    for (uint32_t i = 0; i < NTOT; i++) { fr[i].id = i; fr[i].f = 0; }
    { size_t off = 0;
      for (size_t s = 0; s < tr.nsteps; s++) {
          for (int j = 0; j < tr.nsel[s]; j++) fr[tr.ids[off + j]].f++;
          off += (size_t)tr.nsel[s];
      } }
    qsort(fr, NTOT, sizeof(freq_t), freq_cmp);
    uint16_t *pin = malloc((size_t)NTOT * sizeof(uint16_t));
    for (uint32_t i = 0; i < NTOT; i++) pin[i] = (uint16_t)fr[i].id;
    if (n_pin > NTOT) n_pin = NTOT;

    static const double TGT[] = { 0.90, 0.95, 0.99 };
    const int NT = 3;
    uint32_t cap_hot[3] = {0,0,0}, cap_geo[3] = {0,0,0};

    printf("\n  cap     LRU      HOT(ds4)  GEO(new)   GEOuseful/issued   residentGiB\n");
    printf("  ------  -------  --------  --------   ----------------   -----------\n");

    uint32_t step = cap_step ? cap_step : (NTOT / 24 < 64 ? 64 : NTOT / 24);
    uint32_t cmin = cap_min ? cap_min : (n_pin > top_k ? n_pin : top_k);
    uint32_t cmax = cap_max ? cap_max : NTOT;
    if (cmax > NTOT) cmax = NTOT;
    for (uint32_t cap = cmin; cap <= cmax; cap += step) {
        double hr[3]; uint64_t useful = 0, issued = 0;
        for (int mode = 0; mode < 3; mode++) {
            ds4_prefetch_cache c;
            uint32_t up = (mode == 0) ? 0 : n_pin;
            uint32_t ud = (mode == 2) ? degree : 0;
            const uint16_t *pp = (mode == 0) ? NULL : pin;
            if (cap < up) { hr[mode] = -1; continue; }
            if (!ds4_prefetch_cache_init(&c, NTOT, cap, up, ud, pp)) { hr[mode] = -1; continue; }
            size_t off = 0;
            for (size_t s = 0; s < tr.nsteps; s++) {
                ds4_prefetch_cache_step(&c, (mode == 2) ? &gg : NULL, 0,
                                        tr.ids + off, (uint32_t)tr.nsel[s]);
                off += (size_t)tr.nsel[s];
            }
            hr[mode] = ds4_prefetch_cache_hit_rate(&c);
            if (mode == 2) { useful = c.prefetch_useful; issued = c.prefetch_issued; }
            ds4_prefetch_cache_free(&c);
        }
        for (int i = 0; i < NT; i++) {
            if (hr[1] >= 0 && cap_hot[i] == 0 && hr[1] >= TGT[i]) cap_hot[i] = cap;
            if (hr[2] >= 0 && cap_geo[i] == 0 && hr[2] >= TGT[i]) cap_geo[i] = cap;
        }
        double frac = issued ? 100.0 * (double)useful / (double)issued : 0.0;
        printf("  %6u  %6.2f%%  %6.2f%%  %6.2f%%   %5.1f%% (%llu/%llu)  %8.2f\n",
               cap, 100*hr[0], 100*hr[1], 100*hr[2], frac,
               (unsigned long long)useful, (unsigned long long)issued,
               cap * per_expert_mib / 1024.0);
    }

    printf("\n  Resident experts (= VRAM/RAM) needed for a target hit-rate:\n");
    printf("  target  HOT(ds4)  GEO(new)   saved experts   saved memory\n");
    printf("  ------  --------  --------   -------------   ------------\n");
    for (int i = 0; i < NT; i++) {
        uint32_t h = cap_hot[i], g = cap_geo[i];
        if (!h || !g) { printf("  %4.0f%%   %s\n", 100*TGT[i], "n/a (raise cap range)"); continue; }
        long d = (long)h - (long)g;
        printf("  %4.0f%%   %7u   %7u   %+11ld    %+7.2f GiB\n",
               100*TGT[i], h, g, d, (double)d * per_expert_mib / 1024.0);
    }
    printf("\n  (per-expert size assumed %.1f MiB; pass --per_expert_mib to match\n"
           "   the real routed-expert byte size. Positive saved = less VRAM/RAM.)\n",
           per_expert_mib);

    free(fr); free(pin); free(tr.ids); free(tr.nsel);
    ds4_expert_geometry_free(&gg);
    ds4_expert_geometry_free(&geo);
    return 0;
}
