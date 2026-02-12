# Experiment 06b (FIXED): XCD-Aware GEMM
# ========================================

HIPCC = hipcc
ARCH = gfx942
HIPFLAGS = --offload-arch=$(ARCH) -O3

TARGET = swizzled_gemm
SRC = swizzled_gemm.cpp

.PHONY: all run run-large profile clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPFLAGS) -o $@ $<
	@echo "✓ Built $(TARGET) for $(ARCH)"

run: $(TARGET)
	./$(TARGET) 2048 2048 2048

run-large: $(TARGET)
	@echo "Running large matrix (4096) to see swizzle effect..."
	./$(TARGET) 4096 4096 4096

profile: $(TARGET) metrics.txt
	rocprof -i metrics.txt -o results.csv ./$(TARGET) 4096 4096 4096
	@echo ""
	@echo "Results in results.csv"
	@head -30 results.csv 2>/dev/null || cat results.csv

clean:
	rm -f $(TARGET) *.csv *.json
