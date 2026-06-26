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

#endif /* DS4_EXPERT_TRACE_H */
