/* ds4_expert_trace — see ds4_expert_trace.h.
 *
 * Inert unless DS4_EXPERT_TRACE names an output file. The file is a plain-text
 * trace consumed by expert_prefetch/expert_trace_replay.c:
 *   # ds4 expert trace v1
 *   <layer> <e0> <e1> ... <e_{k-1}>      (one routed selection per line)
 */
#include "ds4_expert_trace.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

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
