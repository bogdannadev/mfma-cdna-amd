# CDNA3/MI300X MFMA Learning Course

A progressive learning path for AMD GPU kernel programming, focusing on Matrix Fused Multiply-Add (MFMA) instructions on MI300X (gfx942).

## Prerequisites

Run these commands on your MI300X droplet:

```bash
# Create project structure
mkdir -p ~/mfma-course/{01-hello-hip,02-wavefront-basics,03-lds-memory,04-mfma-intro,05-mfma-gemm}
cd ~/mfma-course

# Clone the matrix calculator (essential tool for understanding MFMA layouts)
git clone https://github.com/ROCm/amd_matrix_instruction_calculator
pip3 install tabulate --break-system-packages

# Verify calculator works - this lists all MFMA instructions for CDNA3
cd amd_matrix_instruction_calculator
./matrix_calculator.py --architecture cdna3 --list-instructions
cd ..

# Verify ROCm environment
rocminfo | grep -A 5 "Name:.*gfx942"
hipcc --version
```

## Course Structure

| Experiment | Topic | Key Concepts |
|------------|-------|--------------|
| 01 | Hello HIP | Kernel launch, thread indexing, memory management |
| 02 | Wavefront Basics | 64-thread wavefronts, SIMD execution, lane operations |
| 03 | LDS Memory | Shared memory, bank conflicts, synchronisation |
| 04 | MFMA Introduction | Matrix instructions, AGPRs, outer products |
| 05 | MFMA GEMM | Tiled matrix multiply, register blocking |

## Key References

- **MI300 ISA PDF**: Section 7.1 "Matrix fused-multiply-add (MFMA)" (pages 40-55)
- **LLVM gfx942 Instructions**: https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
- **ROCm HIP Guide**: https://rocm.docs.amd.com/projects/HIP/en/latest/
- **AMD Matrix Calculator**: https://github.com/ROCm/amd_matrix_instruction_calculator

## MI300X Key Specifications (from your rocminfo)

```
Device:           AMD Instinct MI300X VF (gfx942)
Compute Units:    304
Wavefront Size:   64 threads
Max Threads/Block: 1024
VRAM:             192 GB (200998912 KB)
L1 Cache:         32 KB
L2 Cache:         4 MB  
L3 Cache:         256 MB
```

## Building Experiments

Each experiment has its own directory. To build:

```bash
cd ~/mfma-course/01-hello-hip
hipcc --offload-arch=gfx942 -O3 -o hello_hip hello_hip.cpp
./hello_hip
```

Or use the provided Makefile:

```bash
cd ~/mfma-course
make -C 01-hello-hip hello_hip
```

## Examining Generated Assembly

To understand what the compiler generates (critical for optimisation):

```bash
hipcc --offload-arch=gfx942 -O3 -S -o kernel.s kernel.cpp
# Or save all intermediate files:
hipcc --offload-arch=gfx942 -O3 --save-temps -o kernel kernel.cpp
```

## Profiling with rocprof

```bash
# Basic profiling
rocprof --stats ./your_kernel

# Detailed counters
rocprof -i counters.txt ./your_kernel

# Where counters.txt contains:
# pmc: GRBM_COUNT, GRBM_GUI_ACTIVE
# pmc: SQ_WAVES, SQ_INSTS_VALU
```
