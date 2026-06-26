# Standalone build for the geometric expert-prefetch module + CPU simulator.
#
# This is intentionally separate from the main ds4 Makefile so that adding this
# experimental, opt-in feature does NOT modify or destabilise the existing build.
# It is CPU-only and uses no GPU.
#
#   make            build ./expert_prefetch_sim
#   make run        build and run the simulator with defaults
#   make clean      remove build outputs

CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -std=c99
LDLIBS ?= -lm

.PHONY: all run trace-replay clean

all: expert_prefetch_sim expert_trace_replay

expert_prefetch_sim: ds4_expert_prefetch.o expert_prefetch_sim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

expert_trace_replay: ds4_expert_prefetch.o expert_trace_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_expert_prefetch.o: ds4_expert_prefetch.c ds4_expert_prefetch.h
	$(CC) $(CFLAGS) -c -o $@ ds4_expert_prefetch.c

expert_prefetch_sim.o: expert_prefetch_sim.c ds4_expert_prefetch.h
	$(CC) $(CFLAGS) -c -o $@ expert_prefetch_sim.c

expert_trace_replay.o: expert_trace_replay.c ds4_expert_prefetch.h
	$(CC) $(CFLAGS) -c -o $@ expert_trace_replay.c

run: expert_prefetch_sim
	./expert_prefetch_sim

trace-replay: expert_trace_replay
	./expert_trace_replay --geo=geometry.ds4geo --synth=2000

clean:
	rm -f expert_prefetch_sim expert_trace_replay *.o
