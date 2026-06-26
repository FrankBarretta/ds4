/* ds4_expert_trace — see ds4_expert_trace.h.
 *
 * Inert unless DS4_EXPERT_TRACE names an output file. The file is a plain-text
 * trace consumed by expert_prefetch/expert_trace_replay.c:
 *   # ds4 expert trace v1
 *   <layer> <e0> <e1> ... <e_{k-1}>      (one routed selection per line)
 */
#include "ds4_expert_trace.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *g_trace_fp;
static int   g_trace_init;
static pthread_mutex_t g_trace_mu = PTHREAD_MUTEX_INITIALIZER;

void ds4_expert_trace_record(unsigned layer, const int *selected,
                             unsigned n_selected) {
    if (!g_trace_init) {
        pthread_mutex_lock(&g_trace_mu);
        if (!g_trace_init) {
            const char *path = getenv("DS4_EXPERT_TRACE");
            if (path && path[0]) {
                g_trace_fp = fopen(path, "w");
                if (g_trace_fp)
                    fprintf(g_trace_fp, "# ds4 expert trace v1\n");
            }
            g_trace_init = 1;
        }
        pthread_mutex_unlock(&g_trace_mu);
    }
    if (!g_trace_fp || !selected) return;

    pthread_mutex_lock(&g_trace_mu);
    fprintf(g_trace_fp, "%u", layer);
    for (unsigned i = 0; i < n_selected; i++)
        fprintf(g_trace_fp, " %d", selected[i]);
    fputc('\n', g_trace_fp);
    fflush(g_trace_fp);            /* flush per line so a killed run keeps data */
    pthread_mutex_unlock(&g_trace_mu);
}

/* ---- opt-in expert prefetch geometry (DS4_EXPERT_PREFETCH=<file.ds4geo>) ---- */

static int               g_pf_init;
static int               g_pf_loaded;
static uint32_t          g_pf_nlayer, g_pf_nexpert, g_pf_degree, g_pf_use_degree;
static uint16_t         *g_pf_neigh;
static pthread_mutex_t   g_pf_mu = PTHREAD_MUTEX_INITIALIZER;

static void ds4_expert_prefetch_lazy_init(void) {
    if (g_pf_init) return;
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_init) { pthread_mutex_unlock(&g_pf_mu); return; }
    g_pf_init = 1;

    const char *path = getenv("DS4_EXPERT_PREFETCH");
    if (path && path[0]) {
        FILE *fp = fopen(path, "rb");
        if (fp) {
            char magic[8];
            uint32_t hdr[4];
            if (fread(magic, 1, 8, fp) == 8 &&
                memcmp(magic, "DS4GEO\0\0", 8) == 0 &&
                fread(hdr, sizeof(uint32_t), 4, fp) == 4 &&
                hdr[0] == 1u && hdr[1] > 0 && hdr[2] > 0 &&
                hdr[2] <= 0xFFFFu && hdr[3] > 0) {
                const uint64_t n = (uint64_t)hdr[1] * hdr[2] * hdr[3];
                uint16_t *nb = (uint16_t *)malloc(n * sizeof(uint16_t));
                if (nb && fread(nb, sizeof(uint16_t), n, fp) == n) {
                    g_pf_nlayer = hdr[1];
                    g_pf_nexpert = hdr[2];
                    g_pf_degree = hdr[3];
                    g_pf_neigh = nb;
                    g_pf_loaded = 1;
                    unsigned ud = g_pf_degree;
                    const char *ds = getenv("DS4_EXPERT_PREFETCH_DEGREE");
                    if (ds && ds[0]) {
                        unsigned v = (unsigned)strtoul(ds, NULL, 10);
                        if (v > 0 && v < ud) ud = v;
                    }
                    g_pf_use_degree = ud;
                    fprintf(stderr,
                            "ds4: expert prefetch enabled: %s "
                            "(%u layers x %u experts x %u neighbours, using degree %u)\n",
                            path, g_pf_nlayer, g_pf_nexpert, g_pf_degree,
                            g_pf_use_degree);
                } else {
                    free(nb);
                }
            }
            fclose(fp);
        }
        if (!g_pf_loaded)
            fprintf(stderr,
                    "ds4: DS4_EXPERT_PREFETCH set but could not load %s\n", path);
    }
    pthread_mutex_unlock(&g_pf_mu);
}

int ds4_expert_prefetch_enabled(void) {
    ds4_expert_prefetch_lazy_init();
    return g_pf_loaded;
}

unsigned ds4_expert_prefetch_collect(unsigned layer, const int *selected,
                                     unsigned n_selected, int *out,
                                     unsigned max) {
    if (!g_pf_loaded || !selected || !out || max == 0) return 0;
    if (layer >= g_pf_nlayer) return 0;
    const uint32_t ne = g_pf_nexpert, deg = g_pf_degree, ud = g_pf_use_degree;
    unsigned count = 0;
    for (unsigned i = 0; i < n_selected; i++) {
        const int s = selected[i];
        if (s < 0 || (uint32_t)s >= ne) continue;
        const uint16_t *nb = g_pf_neigh + ((size_t)layer * ne + (uint32_t)s) * deg;
        for (uint32_t r = 0; r < ud; r++) {
            const int e = (int)nb[r];
            if ((uint32_t)e >= ne) continue;
            int dup = 0;
            for (unsigned j = 0; j < n_selected; j++)
                if (selected[j] == e) { dup = 1; break; }
            for (unsigned j = 0; !dup && j < count; j++)
                if (out[j] == e) { dup = 1; break; }
            if (dup) continue;
            out[count++] = e;
            if (count >= max) return count;
        }
    }
    return count;
}
