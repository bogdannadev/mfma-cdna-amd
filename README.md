# MFMA-CDNA-AMD: A Practical Guide to AMD GPU Kernel Programming

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-6.0%2B-green.svg)](https://rocm.docs.amd.com/)
[![Architecture](https://img.shields.io/badge/Target-gfx942%20(MI300X)-red.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

A progressive, hands-on learning path for AMD GPU kernel programming, focusing on Matrix Fused Multiply-Add (MFMA) instructions on CDNA3 architecture. This guide takes you from your first HIP kernel to understanding the register-level mechanics of matrix operations.

---

## Table of Contents

- [Why This Guide Exists](#why-this-guide-exists)
- [AMD vs NVIDIA: Technical Context](#amd-vs-nvidia-technical-context)
- [Getting Free GPU Access](#getting-free-gpu-access)
- [What You Will Learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Course Structure](#course-structure)
- [Getting Started](#getting-started)
- [Essential Documentation](#essential-documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Licence](#licence)

---

## Why This Guide Exists

Modern AI inference and training workloads are dominated by matrix operations. Understanding how these operations execute at the hardware level—specifically through Matrix Fused Multiply-Add (MFMA) instructions—is essential knowledge for anyone serious about GPU kernel optimisation.

This repository exists because:

1. **MFMA documentation is scattered** — AMD provides excellent ISA manuals, but connecting theory to practice requires working through examples.

2. **Most tutorials target NVIDIA** — The CUDA ecosystem has decades of educational material. AMD's ROCm ecosystem, whilst technically mature, lacks beginner-friendly progressive learning paths.

3. **Kernel programming is a rare skill** — Fewer than a hundred engineers globally possess deep expertise in AMD-specific matrix instruction programming. This guide aims to expand that number.

4. **Open-source inference engines need contributors** — Projects like vLLM, Composable Kernel, and ROCm libraries require developers who understand the underlying hardware.

We encourage you to type every line of code manually rather than copy-pasting. The muscle memory and the errors you encounter along the way are part of the learning process.

---

## AMD vs NVIDIA: Technical Context

This section provides factual technical comparisons to help you understand the AMD ecosystem. We present these as engineering considerations, not product recommendations.

### Architectural Differences

| Aspect | AMD CDNA3 (MI300X) | NVIDIA Hopper (H100) |
|--------|-------------------|---------------------|
| Wavefront/Warp Size | 64 threads | 32 threads |
| Matrix Unit | MFMA (Matrix Core) | Tensor Core |
| Shared Memory Term | LDS (Local Data Share) | Shared Memory |
| Register File | VGPR + AGPR (Accumulation) | Unified Register File |
| Memory | 192 GB HBM3 | 80 GB HBM3 |
| Compute Dies | 8 XCDs per package | Monolithic |

### Why Learn AMD Kernel Programming?

**1. Open-Source Software Stack**

ROCm is fully open-source from the compiler (LLVM-based) through runtime libraries. You can read, modify, and understand every layer of the stack. This transparency is invaluable for learning and debugging.

```bash
# You can inspect generated assembly directly
hipcc --offload-arch=gfx942 -S -o kernel.s kernel.cpp
```

**2. Growing Deployment Footprint**

AMD Instinct GPUs power several of the world's largest supercomputers (Frontier, LUMI, El Capitan). Enterprise adoption is expanding, creating demand for developers with AMD-specific expertise.

**3. Transferable Concepts**

The fundamental concepts—wavefront execution, memory coalescing, occupancy optimisation, matrix tiling—transfer between vendors. Learning on AMD makes you a better GPU programmer overall.

**4. HIP Portability**

HIP (Heterogeneous-Compute Interface for Portability) code can compile for both AMD and NVIDIA GPUs. Skills learned here apply broadly:

```cpp
// This kernel compiles for both gfx942 (AMD) and sm_90 (NVIDIA)
__global__ void vector_add(float* c, const float* a, const float* b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}
```

**5. Composable Kernel Expertise**

AMD's Composable Kernel (CK) library is the foundation for high-performance GEMM and attention kernels in the ROCm ecosystem. Understanding MFMA is prerequisite knowledge for contributing to CK.

### Honest Limitations

- **Ecosystem maturity**: CUDA has broader library support and more community resources.
- **Tooling**: NVIDIA's profiling tools (Nsight) are more polished than ROCm's rocprof.
- **Documentation**: AMD documentation, whilst comprehensive, can be harder to navigate.

These limitations are improving rapidly, but awareness helps set realistic expectations.

---

## Getting Free GPU Access

You do not need to purchase hardware to learn AMD kernel programming.

### Option 1: AMD AI Developer Program (Recommended)

The AMD AI Developer Program provides the most comprehensive free access package.

**What You Get:**
- **$100 in DigitalOcean credits** for GPU instances
- Access to private Discord channel with AMD engineers
- Monthly hardware sweepstakes (Radeon GPUs, Ryzen AI PCs)
- Early access to developer events and workshops

**How to Join:**

1. Navigate to [amd.com/en/developer/ai-dev-program.html](https://www.amd.com/en/developer/ai-dev-program.html)
2. Click "Join Now" and complete the registration form
3. After registration, you receive an email with a link to the member site
4. On the member site, find and click the "$100 credit link" to activate your DigitalOcean credits
5. Create or log into your DigitalOcean account to receive the credits

**Important Notes:**
- Credits are activated only after you create/log into the cloud account via the member site link
- For questions: `ai_dev_program@amd.com`

### Maximising Your Free Credits

Here are strategies to make your free GPU hours last:

1. **Use snapshots** — Save VM state before destroying to preserve environment setup
2. **Destroy, don't power off** — Powered-off instances still consume credits
3. **Batch your GPU work** — Plan sessions to maximise productive time
4. **Start with 1× GPU** — The 8× MI300X configuration consumes credits 8× faster

---

## What You Will Learn

This course builds your understanding progressively, from basic concepts to advanced matrix operations.

### Skills Acquired

**Foundation Level:**
- HIP kernel syntax and compilation
- GPU thread hierarchy (grids, blocks, threads)
- Memory management (allocation, transfer, synchronisation)
- Error handling patterns for GPU code

**Intermediate Level:**
- Wavefront execution model (64-thread SIMD)
- Register types: VGPR (vector), SGPR (scalar), AGPR (accumulation)
- Local Data Share (LDS) usage and bank conflict avoidance
- Cross-lane communication via shuffle operations

**Advanced Level:**
- MFMA instruction mechanics and register layouts
- Matrix tiling strategies for GEMM
- Assembly inspection and optimisation
- Profiling with rocprof

### Composable Kernel Connection

Each experiment in this course relates directly to patterns used in AMD's Composable Kernel library:

| This Course | Composable Kernel Equivalent |
|-------------|------------------------------|
| Thread indexing | `block_2_etile_op` coordinate transforms |
| LDS management | `lds_buffer` abstractions |
| Bank conflict avoidance | `lds_direct_load` patterns |
| MFMA intrinsics | `mfma_op` wrappers |
| Register blocking | `block_gemm_pipeline` |

Understanding these fundamentals positions you to read, understand, and contribute to CK.

---

## Prerequisites

### Knowledge Requirements

- **C++ fundamentals** — Classes, templates, pointers, memory management
- **Basic linear algebra** — Matrix multiplication, transpose operations
- **Command line comfort** — SSH, bash, file navigation
- **Optional but helpful** — Any prior GPU programming experience (CUDA, OpenCL)

### Software Requirements

On your local machine (for connecting to remote GPU):
- SSH client
- Text editor or IDE with remote development support (VSCode recommended)
- Git

On the GPU instance (pre-installed in DigitalOcean AMD GPU images):
- ROCm 6.0 or later
- HIP compiler (hipcc)

---

## Hardware Requirements

### Primary Target: AMD Instinct MI300X (gfx942)

This course is developed and tested on MI300X. Key specifications:

```
Device:           AMD Instinct MI300X
Architecture:     CDNA3 (gfx942)
Compute Units:    304
Wavefront Size:   64 threads
Max Threads/Block: 1024
VRAM:             192 GB HBM3
L1 Cache:         32 KB per CU
L2 Cache:         32 MB (4 MB per XCD × 8)
Matrix Cores:     MFMA v3
```

### Architecture Transferability

The concepts taught here transfer to other AMD architectures:

| Architecture | Example GPUs | MFMA Support | Notes |
|--------------|--------------|--------------|-------|
| CDNA3 | MI300X, MI300A | Full (v3) | Primary target |
| CDNA2 | MI250X, MI210 | Full (v2) | Minor instruction differences |
| CDNA1 | MI100 | Full (v1) | Older but compatible |
| RDNA3 | RX 7900 XTX | WMMA only | Different matrix instructions |
| GCN | RX 580, Vega | None | No matrix acceleration |

If you're using MI250X or MI210, most examples work with minimal changes (primarily the `--offload-arch` flag).

---

## Course Structure

The course consists of progressive experiments. Complete them in order—each builds upon concepts from previous experiments.

| Experiment | Topic | Duration | Key Concepts |
|------------|-------|----------|--------------|
| 01 | Hello HIP | 1–2 hours | Kernel launch, thread indexing, memory management |
| 02 | Wavefront Basics | 2–3 hours | 64-thread SIMD, lane operations, divergence |
| 03 | LDS Memory | 2–3 hours | Shared memory, bank conflicts, synchronisation |
| 04 | MFMA Introduction | 2–3 hours | Matrix cores, AGPR/VGPR, correct vector types |
| 05 | MFMA GEMM | 2–3 hours | Tiled GEMM, cooperative loading, optimisation |

### Experiment Format

Each experiment follows a consistent structure:

```
XX-experiment-name/
├── experiment_name.cpp    # Heavily commented source code
└── README.md              # Theory explanation and exercises
```

The source files contain extensive educational comments explaining:
- Why each line exists
- How it relates to hardware behaviour
- Common mistakes and how to avoid them
- Connections to Composable Kernel patterns

---

## Getting Started

### Step 1: Access a GPU Instance

Follow the [Getting Free GPU Access](#getting-free-gpu-access) section to obtain an MI300X instance.

Once connected via SSH, verify your environment:

```bash
# Check GPU is visible
rocminfo | grep -E "Name:.*gfx"

# Expected output includes:
#   Name:                    gfx942

# Verify compiler
hipcc --version

# Expected: HIP version with ROCm path
```

### Step 2: Clone This Repository

```bash
cd ~
git clone https://github.com/bogdannadev/mfma-cdna-amd.git
cd mfma-cdna-amd
```

### Step 3: Build and Run Your First Experiment

```bash
cd 01-hello-hip

# Build the experiment
hipcc --offload-arch=gfx942 -O3 -o hello_hip hello_hip.cpp

# Run it
./hello_hip
```

You should see device information and a simple computation result.

### Step 4: Study the Code

Open `hello_hip.cpp` in your editor and read through the comments carefully. The comments explain:
- Why we use `__global__` for kernel functions
- How thread indexing works
- Why bounds checking is essential
- How memory transfers between host and device work

**We strongly encourage typing the code yourself** rather than running the pre-written version. The act of typing, making mistakes, and debugging builds deeper understanding.

### Using the Makefile

For convenience, a Makefile is provided:

```bash
# From repository root
make help          # Show available targets
make exp01         # Build experiment 01
make exp02         # Build experiment 02
make exp04         # Build experiment 04 (MFMA intro)
make exp05         # Build experiment 05 (MFMA GEMM)
make all           # Build all experiments
make run           # Build and run all experiments
make clean         # Remove built binaries
```

### Debug Builds

Debug builds compile with `-O0 -g -save-temps` flags, producing unoptimised binaries with debug symbols and all intermediate files:

```bash
make exp01-debug   # Creates 01-hello-hip/hello_hip_debug
make exp04-debug   # Creates 04-mfma-intro/mfma_intro_debug
make exp05-debug   # Creates 05-mfma-gemm/mfma_gemm_debug
```

The `-save-temps` flag preserves intermediate files (`.ll`, `.bc`, `.o`) which are useful for understanding the compilation pipeline.

### Assembly Generation

Generate human-readable assembly to inspect what the compiler produces:

```bash
make exp01-asm     # Creates 01-hello-hip/hello_hip.s
make exp04-asm     # Creates 04-mfma-intro/mfma_intro.s
make exp05-asm     # Creates 05-mfma-gemm/mfma_gemm.s
```

In the generated `.s` files, look for MFMA instructions:

```asm
v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]
```

This is essential for verifying correct code generation and understanding register allocation.

### Profiling Your Kernels

Use rocprof to measure kernel performance:

```bash
# Basic timing statistics
rocprof --stats ./mfma_gemm

# Detailed hardware counters
echo "pmc: SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_MFMA" > counters.txt
rocprof -i counters.txt ./mfma_gemm
```

---

## Essential Documentation

**AMD Instinct MI300 ISA Manual**

The authoritative reference for CDNA3 instructions. Section 7.1 covers MFMA in detail.

📥 Download: [AMD Instinct MI300 Series ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)

Key sections for this course:
- Section 2: Program Organisation (thread model)
- Section 4: Kernel State (registers, program counter)
- Section 5: Scalar ALU Operations
- Section 6: Vector ALU Operations
- Section 7: Matrix Fused Multiply-Add (MFMA)
- Section 9: Data Share Operations (LDS)

---

## Roadmap

### Currently Available

| Status | Experiment | Description |
|--------|------------|-------------|
| ✅ | 01-hello-hip | HIP fundamentals, kernel launch, memory management |
| ✅ | 02-wavefront-basics | Wavefront execution, lane operations, divergence |
| ✅ | 03-lds-memory | Local Data Share, bank conflicts, synchronisation |
| ✅ | 04-mfma-intro | MFMA instruction basics, AGPR usage, correct vector types |
| ✅ | 05-mfma-gemm | Tiled GEMM implementation with MFMA |

### Planned Additions

These experiments are under development:

- **06-mfma-attention** — Flash Attention kernel using MFMA
- **07-multi-gpu** — Peer-to-peer communication patterns
- **08-profiling-deep-dive** — Advanced rocprof usage and optimisation
- **09-composable-kernel-study** — Guided tour of CK source code

Watch this repository for updates.

---

## Contributing

Contributions are welcome. Areas where help is particularly valuable:

- **Bug fixes** — If you find errors in code or documentation
- **Clarity improvements** — If explanations are confusing, suggest improvements
- **Additional examples** — Small, focused examples that illustrate specific concepts
- **Architecture ports** — Testing/adapting examples for MI250X, MI210

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improved-explanation`)
3. Make your changes with clear commit messages
4. Ensure code compiles and runs correctly
5. Submit a pull request with description of changes

---

## Licence

This project is licensed under the Apache 2.0 Licence. See `LICENSE` file for details.

---

## Acknowledgements

- AMD AI Developer Program for GPU access credits
- ROCm team for comprehensive documentation
- Composable Kernel project for demonstrating production patterns