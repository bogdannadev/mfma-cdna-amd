# =============================================================================
# MFMA Course Makefile
# =============================================================================
#
# Build all experiments: make all
# Build specific:        make exp01 / make exp02 / etc.
# Clean:                 make clean
# Run all:               make run
#
# =============================================================================

HIPCC = hipcc
ARCH = gfx942
HIPFLAGS = --offload-arch=$(ARCH) -O3
DEBUGFLAGS = --offload-arch=$(ARCH) -O0 -g -save-temps

# Experiment binaries
EXP01 = 01-hello-hip/hello_hip
EXP02 = 02-wavefront-basics/wavefront_basics
EXP03 = 03-lds-memory/lds_memory
EXP04 = 04-mfma-intro/mfma_intro
EXP05 = 05-mfma-gemm/mfma_gemm

ALL_EXPS = $(EXP01) $(EXP02) $(EXP03) $(EXP04) $(EXP05)

.PHONY: all clean run exp01 exp02 exp03 exp04 exp05 help

help:
	@echo "MFMA Course Build System"
	@echo "========================"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all experiments"
	@echo "  exp01   - Build Hello HIP"
	@echo "  exp02   - Build Wavefront Basics"
	@echo "  exp03   - Build LDS Memory"
	@echo "  exp04   - Build MFMA Introduction"
	@echo "  exp05   - Build MFMA GEMM"
	@echo "  run     - Build and run all experiments"
	@echo "  clean   - Remove all binaries"
	@echo ""
	@echo "Debug builds (saves .s assembly files):"
	@echo "  make exp01-debug"
	@echo ""

all: $(ALL_EXPS)
	@echo "✓ All experiments built successfully"

# Individual experiments
exp01: $(EXP01)
exp02: $(EXP02)
exp03: $(EXP03)
exp04: $(EXP04)
exp05: $(EXP05)

# Build rules
$(EXP01): 01-hello-hip/hello_hip.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

$(EXP02): 02-wavefront-basics/wavefront_basics.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

$(EXP03): 03-lds-memory/lds_memory.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

$(EXP04): 04-mfma-intro/mfma_intro.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

$(EXP05): 05-mfma-gemm/mfma_gemm.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# Debug builds (with assembly output)
%-debug: %.cpp
	$(HIPCC) $(DEBUGFLAGS) -o $@ $<
	@echo "Assembly saved to: $*.s"

# Run all experiments in order
run: all
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "             Running MFMA Course Experiments"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	./$(EXP01)
	@echo "────────────────────────────────────────────────────────────────"
	./$(EXP02)
	@echo "────────────────────────────────────────────────────────────────"
	./$(EXP03)
	@echo "────────────────────────────────────────────────────────────────"
	./$(EXP04)
	@echo "────────────────────────────────────────────────────────────────"
	./$(EXP05)
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "             All experiments completed!"
	@echo "════════════════════════════════════════════════════════════════"

clean:
	rm -f $(ALL_EXPS)
	rm -f */*.s */*.ll */*.bc */*.o
	rm -f */*-debug
	@echo "✓ Cleaned all build artifacts"