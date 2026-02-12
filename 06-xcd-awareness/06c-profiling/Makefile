# =============================================================================
# Experiment 06c: L2 Cache Profiling
# =============================================================================

HIPCC = hipcc
ARCH = gfx942
HIPFLAGS = --offload-arch=$(ARCH) -O3

TARGET = l2_profiling
SRC = l2_profiling.cpp

.PHONY: all run profile profile-detailed clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPFLAGS) -o $@ $<
	@echo "✓ Built $(TARGET) for $(ARCH)"

run: $(TARGET)
	./$(TARGET)

# Basic profiling with L2 cache metrics
# Grundläggande profilering med L2-cache-metriker
profile: $(TARGET) metrics.txt
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║  Profiling L2 Cache Hit Rates                                 ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	rocprof -i metrics.txt -o results.csv ./$(TARGET)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Results / Resultat (from results.csv):"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@cat results.csv 2>/dev/null || echo "Check results.csv"
	@echo ""
	@echo "Look for L2CacheHit values for k_attention_like_access kernel."
	@echo "NAIVE (use_swizzle=0) should have LOWER L2CacheHit than SWIZZLED (use_swizzle=1)."

# Detailed profiling with rocprofv3
# Detaljerad profilering med rocprofv3
profile-v3: $(TARGET)
	@echo ""
	@echo "Profiling with rocprofv3 (trace files)..."
	@echo ""
	rocprofv3 --hip-trace --kernel-trace -o trace_results ./$(TARGET)
	@echo ""
	@echo "Results in trace_results/"
	@echo "View with Perfetto: https://ui.perfetto.dev/"

clean:
	rm -f $(TARGET) *.csv *.json
	rm -rf trace_results/
