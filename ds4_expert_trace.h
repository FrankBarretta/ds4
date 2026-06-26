/* ds4_expert_trace — opt-in capture of routed expert selections for offline
 * prefetch analysis (see expert_prefetch/).
 *
 * This is inert by default: ds4_expert_trace_record() does nothing unless the
 * environment variable DS4_EXPERT_TRACE names an output file. When the variable
 * is unset, ds4 behaviour is byte-for-byte identical to before this file. */
#ifndef DS4_EXPERT_TRACE_H
#define DS4_EXPERT_TRACE_H

/* Record one routed selection: `selected[0..n_selected-1]` are the expert ids
 * chosen at MoE layer `layer` for the current token, in decode order. */
void ds4_expert_trace_record(unsigned layer, const int *selected,
                             unsigned n_selected);

/* ---- opt-in expert prefetch (geometry-driven streaming cache warming) ----
 *
 * Enabled only when the environment variable DS4_EXPERT_PREFETCH names a
 * .ds4geo neighbour table (built by expert_prefetch/). Inert otherwise.
 * Optional DS4_EXPERT_PREFETCH_DEGREE caps how many neighbours per expert. */

/* True once a geometry has been loaded (lazy, from the env var on first call). */
int ds4_expert_prefetch_enabled(void);

/* Given the experts selected at `layer`, write the deduped neighbour ids to
 * prefetch into `out` (capacity `max`), excluding the already-selected experts.
 * Returns the count written (0 when disabled). */
unsigned ds4_expert_prefetch_collect(unsigned layer, const int *selected,
                                     unsigned n_selected, int *out, unsigned max);

#endif /* DS4_EXPERT_TRACE_H */
