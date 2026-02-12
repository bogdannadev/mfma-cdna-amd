# =============================================================================
# Experiment 06a: XCD Discovery — Makefile
# =============================================================================
#
# BUILD TARGETS — BYGGMÅL:
#   make            — Build the discovery tool / Bygg upptäcktsverktyget
#   make run        — Build and run / Bygg och kör
#   make profile    — Run with rocprof L2 cache metrics / Kör med rocprof L2-cache-mätvärden
#   make clean      — Remove build artifacts / Ta bort byggartefakter
#
# =============================================================================

HIPCC = hipcc
ARCH = gfx942
HIPFLAGS = --offload-arch=$(ARCH) -O3
DEBUGFLAGS = --offload-arch=$(ARCH) -O0 -g -save-temps

TARGET = xcd_discovery
SRC = xcd_discovery.cpp

.PHONY: all run profile profile-v3 clean help

help:
	@echo "Experiment 06a: XCD Discovery"
	@echo "=============================="
	@echo ""
	@echo "Targets:"
	@echo "  make          — Build the discovery tool"
	@echo "  make run      — Build and run"
	@echo "  make profile  — Run with rocprof (L2 cache metrics)"
	@echo "  make profile-v3 — Run with rocprofv3 (newer profiler)"
	@echo "  make debug    — Build with debug symbols"
	@echo "  make clean    — Remove build artifacts"
	@echo ""

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPFLAGS) -o $@ $<
	@echo "✓ Built $(TARGET) for $(ARCH)"

run: $(TARGET)
	@echo ""
	@echo "Running XCD Discovery..."
	@echo "Kör XCD-upptäckt..."
	@echo ""
	./$(TARGET)

# Profile with rocprof v1/v2 — Profilera med rocprof v1/v2
profile: $(TARGET) metrics.txt
	@echo ""
	@echo "Profiling with rocprof (measuring L2 cache metrics)..."
	@echo "Profilerar med rocprof (mäter L2-cache-mätvärden)..."
	@echo ""
	rocprof -i metrics.txt -o results.csv ./$(TARGET)
	@echo ""
	@echo "Results saved to results.csv"
	@echo "Resultat sparade i results.csv"
	@echo ""
	@cat results.csv 2>/dev/null || echo "No results file generated"

# Profile with rocprofv3 — Profilera med rocprofv3
profile-v3: $(TARGET)
	@echo ""
	@echo "Profiling with rocprofv3..."
	@echo "Profilerar med rocprofv3..."
	@echo ""
	rocprofv3 --hip-trace --kernel-trace -o results_v3 ./$(TARGET)
	@echo ""
	@echo "Results saved to results_v3/"
	@ls -la results_v3/ 2>/dev/null || echo "Check results_v3 directory"

# Debug build — Debugbygge
debug: $(SRC)
	$(HIPCC) $(DEBUGFLAGS) -o $(TARGET)_debug $<
	@echo "✓ Built $(TARGET)_debug with debug symbols"

clean:
	rm -f $(TARGET) $(TARGET)_debug
	rm -f *.csv *.txt.tmp *.json
	rm -rf results_v3/
	rm -f *.ll *.bc *.s *.o
	@echo "✓ Cleaned build artifacts"
