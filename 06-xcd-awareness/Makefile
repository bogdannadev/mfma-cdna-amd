# Experiment 06a: XCD Discovery

# BUILD TARGETS:
# make - Build the discovery tool
# make run - Build and run
# make profile - Run with rocprof L2 cache metrics
# make clean - Remove build artifacts

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
	@echo "Built $(TARGET) for $(ARCH)"

run: $(TARGET)
	@echo ""
	@echo "Running XCD Discovery..."
	@echo ""
	./$(TARGET)

# Profile with rocprof v1/v2
profile: $(TARGET) metrics.txt
	@echo ""
	@echo "Profiling with rocprof (measuring L2 cache metrics)..."
	@echo ""
	rocprof -i metrics.txt -o results.csv ./$(TARGET)
	@echo ""
	@echo "Results saved to results.csv"
	@echo ""
	@cat results.csv 2>/dev/null || echo "No results file generated"

profile-v3: $(TARGET)
	@echo ""
	@echo "Profiling with rocprofv3..."
	@echo ""
	rocprofv3 --hip-trace --kernel-trace -o results_v3 ./$(TARGET)
	@echo ""
	@echo "Results saved to results_v3"
	@ls -la results_v3/ 2>/dev/null || echo "Check results_v3 directory"

debug:
	$(HIPCC) $(DEBUGFLAGS) -o $(TARGET)_debug $<
	@echo "Built $(TARGET)_debug with debug symbols"

clean:
	rm -rf $(TARGET) $(TARGET)_debug
	rm -rf *.csv *.txt.tmp *.json
	rm -rf results_v3/
	rm -rf *.ll *.bc *.s *.offload
	@echo "Cleaned build artifacts"