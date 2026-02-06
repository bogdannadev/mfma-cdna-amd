# HIP Kernel Development — Project Instructions & Reference

> Projektinstruktioner för AMD GPU-kärnutveckling
> (Project instructions for AMD GPU kernel development)
>
> This document serves as the authoritative reference for code conventions,
> best practices, and resource links across all experiments in the MFMA course.
> Add this to your Claude Project knowledge for consistent guidance.

---

## 1. C++ Standard & Compiler Configuration

### Target Standard: C++17

HIP (via `hipcc`, which wraps clang++) supports C++03 through C++20.
We use **C++17** as the project standard — it provides `constexpr if`,
structured bindings, fold expressions, and `std::optional`/`std::variant`
on the host side, all of which are useful for kernel infrastructure code.

**Makefile flag:**
```
HIPFLAGS += -std=c++17
```

**CMake:**
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

### C++20 — Partial Support

All C++20 language features work in HIP **except**:
- Coroutines (not supported on device)
- Modules (not supported)

`consteval` functions work and can be called from both host and device.
The spaceship operator (`<=>`) works in both host and device code.

---

## 2. HIP vs Standard C++ — Key Deviations

### The Dual-Compilation Model

`hipcc` compiles every `.cpp` / `.hip` file **twice**:
1. **Device pass** — compiles `__global__` and `__device__` functions to GPU ISA
2. **Host pass** — compiles host code as standard C++, replacing kernel bodies with dispatch stubs

The `<<<>>>` triple-chevron syntax is **not valid C++**. It is a compiler extension
transformed into `hipLaunchKernelGGL` macro calls during preprocessing.

### Three Code Zones (Mental Model)

When reading any HIP source file, mentally classify every function:

| Zone | Qualifier | Standard Library | Exceptions | Templates | Use Case |
|------|-----------|-----------------|------------|-----------|----------|
| **Host** | `__host__` (implicit default) | Full | Yes | Yes | Resource management, orchestration |
| **Device** | `__global__`, `__device__` | **None** | **No** | Yes | GPU computation |
| **Dual** | `__host__ __device__`, `constexpr` | **None** (must work on both) | **No** | Yes | Utility functions, math helpers |

### Device Code Restrictions

These C++ features are **forbidden** in `__global__` and `__device__` functions:

- `std::` containers (`vector`, `string`, `map`, etc.)
- Exceptions (`throw`, `try`, `catch`)
- RTTI (`dynamic_cast`, `typeid`)
- Virtual functions / dynamic dispatch
- `new` / `delete` (device-side dynamic allocation)
- `long double` parameters (size differs between host and device)
- `va_list` / variadic C-style arguments
- `std::initializer_list` as kernel parameters
- References as kernel parameters (values and pointers only)
- Recursive functions (no call stack on GPU)
- Coroutines and modules (C++20)

### Device Code — What Works

- Templates (extensively used in Composable Kernel)
- `constexpr` functions (automatically `__host__ __device__`)
- `if constexpr` (C++17) — essential for GPU metaprogramming
- Lambdas (with correct capture)
- Structured bindings (C++17)
- Fold expressions (C++17)
- `auto` return type deduction
- Class types with trivial layout as kernel parameters
- `__restrict__` qualifier on pointers

### Kernel Launch Syntax

**Triple-chevron (common):**
```cpp
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
//       ^blocks   ^threads  ^LDS bytes ^hipStream_t
// sharedMem defaults to 0, stream defaults to default stream
```

**hipLaunchKernelGGL (portable alternative):**
```cpp
hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, args...);
```

Both forms are equivalent. Use `<<<>>>` for readability in educational code.

---

## 3. Code Quality Conventions

### Memory Allocation — Host Side

| Pattern | Function | Free Function | Use Case |
|---------|----------|---------------|----------|
| **Pinned** (preferred) | `hipHostMalloc(&ptr, size)` | `hipHostFree(ptr)` | ~3× faster transfers, DMA-friendly |
| **Pageable** (avoid) | `malloc(size)` | `free(ptr)` | Double-copy overhead, OS can swap |
| **Managed** (prototypes) | `hipMallocManaged(&ptr, size)` | `hipFree(ptr)` | Single pointer, auto page migration |
| **Device** | `hipMalloc(&ptr, size)` | `hipFree(ptr)` | GPU-only memory (HBM) |

**Rule: Never use `malloc`/`free` for host buffers that will be transferred to/from GPU.**
Always use `hipHostMalloc` / `hipHostFree` for host-side allocations involved in transfers.

### Headers

Use C++ headers, not C headers:
```cpp
// Correct (C++)        // Avoid (C)
#include <cstdio>       // not <stdio.h>
#include <cstdlib>      // not <stdlib.h>
#include <cstring>      // not <string.h>
#include <cmath>        // not <math.h>
#include <cstdint>      // not <stdint.h>
```

### Naming Conventions

Follow AMD ecosystem conventions:
```cpp
// Pointer prefixes
float* h_input;     // h_ = host memory
float* d_input;     // d_ = device memory

// Constants
constexpr int kBlockSize = 256;    // k prefix for constants
constexpr int kWavefrontSize = 64; // MI300X wavefront size

// Size calculations
size_t bytes = N * sizeof(float);  // Always use size_t for byte counts

// Types
uint32_t n;  // Prefer explicit-width types for kernel parameters
```

### Kernel Signatures

```cpp
__global__ void my_kernel(
    float* __restrict__ output,       // __restrict__ enables compiler optimisations
    const float* __restrict__ input,  // const correctness on input pointers
    const uint32_t n                  // const on value parameters, uint32_t saves registers
) {
    // ...
}
```

### Error Handling

```cpp
#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP Error: %s at %s:%d\n",                   \
                    hipGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
```

Note: `fprintf(stderr, ...)` not `printf(...)` — errors go to stderr.

### Launch Configuration

```cpp
constexpr uint32_t kBlockSize = 256;  // Must be multiple of 64 (wavefront size)

// Ceiling division utility — use everywhere
constexpr auto div_ceil(auto a, auto b) { return (a + b - 1) / b; }

uint32_t grid_size = div_ceil(N, kBlockSize);
kernel<<<grid_size, kBlockSize>>>(d_output, d_input, N);
```

---

## 4. MI300X Hardware Quick Reference

```
Architecture:       CDNA3 (gfx942)
Compute Units:      304 (38 per XCD × 8 XCDs)
Wavefront Size:     64 threads
Max Threads/Block:  1024
VRAM:               192 GB HBM3
HBM Bandwidth:      5.3 TB/s
L1 Cache:           16 KB per CU
L2 Cache:           32 MB total (4 MB per XCD × 8)
LDS per CU:         64 KB
VGPRs per CU:       512 × 32-bit
AGPRs per CU:       512 × 32-bit (accumulation registers for MFMA)
SGPRs per CU:       106 × 32-bit
Matrix Cores:       MFMA v3
PCIe:               Gen5 x16 (~64 GB/s theoretical)
```

---

## 5. Primary Documentation Sources

### Tier 1 — HIP Language & API (Consult First)

| Resource | URL | Use When |
|----------|-----|----------|
| HIP C++ Language Extensions | `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html` | Qualifiers, built-ins, vector types, atomics |
| HIP C++ Language Support | `https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_support.html` | Which C++ features work on device |
| HIP Performance Guidelines | `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html` | Memory alignment, coalescing, occupancy |
| HIP Memory Management | `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html` | All allocation types and their trade-offs |
| HIP Runtime API Reference | `https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api_reference.html` | Function signatures and parameters |

### Tier 2 — Real-World Code Examples

| Resource | URL | Use When |
|----------|-----|----------|
| ROCm Examples (official) | `https://github.com/ROCm/rocm-examples` | HIP-Basic, Tutorials, Performance-Optimization |
| HIP Tutorials (docs) | `https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/saxpy.html` | SAXPY, reduction, programming patterns |
| GPU Programming Patterns | `https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/programming-patterns.html` | Matrix multiply, atomics, stencils, multi-kernel |

**Key files in rocm-examples to study:**
- `HIP-Basic/hello_world/main.hip` — compare with experiment 01
- `HIP-Basic/device_query/` — device properties patterns
- `HIP-Basic/matrix_transpose/` — LDS usage patterns
- `HIP-Basic/shared_memory/` — bank conflict patterns
- `HIP-Basic/assembly_to_executable/` — full ISA → executable chain
- `Tutorials/tiling_matrix_multiply/` — naive → LDS tiling → register tiling
- `Tutorials/tiling_matrix_transpose/` — shared memory coalescing patterns

### Tier 3 — Composable Kernel (CK)

| Resource | URL | Use When |
|----------|-----|----------|
| CK Repository | `https://github.com/ROCm/composable_kernel` | Source code, examples, build instructions |
| CK Wrapper Examples | `.../client_example/25_wrapper/` | Simplest CK API usage (basic GEMM) |
| CK Tile Copy Kernel | `.../example/ck_tile/` (copy kernel) | Simplest ck_tile example for beginners |
| CK Tile Toy GEMM | `.../example/ck_tile/99_toy_example/02_gemm` | GEMM with debugging (toy_example branch) |
| CK Tile Concepts | `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/` | Coordinate transforms, distributed tensors |
| CK Tile README | `.../include/ck_tile/README.md` | Architecture overview, component structure |
| MIOpen Wiki — CK Notes | `https://github.com/ROCm/MIOpen/wiki/Notes-on-Composable-Kernel-library` | Understanding the three CK libraries |
| CK GEMM Debugging Blog | `https://rocm.blogs.amd.com/software-tools-optimization/rocgdb-ck-tile/README.html` | rocgdb debugging, tensor distribution bugs |

### Tier 4 — ISA & Architecture

| Resource | URL | Use When |
|----------|-----|----------|
| MI300 ISA Manual | Project knowledge (PDF) | MFMA instructions, register layout, memory ops |
| AMD Matrix Instruction Calculator | AMD developer tools | MFMA register mapping visualisation |
| libhipcxx | `https://github.com/ROCm/libhipcxx` | C++ standard library primitives for device code |

---

## 6. Composable Kernel Patterns to Recognise

As you progress through experiments, map each concept to its CK equivalent:

| This Course (Experiment) | CK Pattern | CK Location |
|--------------------------|------------|-------------|
| Thread indexing (01) | `block_2_etile_op` | Coordinate transforms |
| Wavefront mechanics (02) | Wave-level primitives | `ck_tile/core/` |
| LDS management (03) | `lds_buffer`, `lds_direct_load` | Tile operators |
| Bank conflict avoidance (03) | XOR-based swizzle | `ck_tile/core/` |
| MFMA intrinsics (04) | `mfma_op` wrappers | `ck_tile/ops/` |
| Register blocking (05) | `block_gemm_pipeline` | Pipeline abstractions |
| XCD-aware swizzling (06) | Workgroup swizzle | Grid-to-XCD mapping |

### CK's Three Libraries (Historical Context)

1. **Legacy CK (static)** — original, compiled per-problem-size, now `legacy_composable_kernel` in MIOpen
2. **CK (dynamic)** — generic kernels, external repository, the main CK library
3. **CK Tile** — newest, tile-based programming model with coordinate transforms, under `include/ck_tile/`

When contributing, target **CK Tile** — it's the active development focus.

---

## 7. Profiling Quick Reference

```bash
# Basic kernel timing
rocprof --stats ./binary

# Hardware counters
echo "pmc: SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_MFMA" > counters.txt
rocprof -i counters.txt ./binary

# L2 cache hit rate (XCD-aware analysis)
echo "pmc: TCP_TCC_READ_REQ_sum, TCP_TCC_HIT_sum" > l2_counters.txt
rocprof -i l2_counters.txt ./binary

# Generate assembly for inspection
hipcc --offload-arch=gfx942 -O3 -S -o kernel.s kernel.cpp
# Look for: v_mfma_f32_16x16x16_f16, ds_read_b128, global_load_dwordx4
```

---

## 8. Build System Reference

### Makefile Pattern
```makefile
HIPCC     = hipcc
ARCH      = gfx942
HIPFLAGS  = --offload-arch=$(ARCH) -O3 -std=c++17
DBGFLAGS  = --offload-arch=$(ARCH) -O0 -g -save-temps -std=c++17
```

### CMake Pattern
```cmake
cmake_minimum_required(VERSION 3.21)
project(experiment LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_ARCHITECTURES gfx942)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
```

---

## 9. Common Anti-Patterns to Avoid

| Anti-Pattern | Correct Pattern | Why |
|--------------|-----------------|-----|
| `malloc` for host buffers | `hipHostMalloc` | Pinned memory avoids staging copies |
| `free` for pinned memory | `hipHostFree` | Mismatched allocator/deallocator = UB |
| `printf` in error macro | `fprintf(stderr, ...)` | Errors belong on stderr |
| `int n` in kernel params | `uint32_t n` | Saves registers, no negative sizes |
| Magic numbers for block size | `constexpr uint32_t kBlockSize` | Compiler optimises better, documents intent |
| Missing `__restrict__` | Always use on kernel pointers | Enables aliasing optimisations |
| `<stdio.h>` | `<cstdio>` | Proper C++ namespace, overload resolution |
| Implicit C++ standard | `-std=c++17` explicit | Reproducible builds across toolchains |
| Manual bit packing for MFMA | `ext_vector_type` attribute | Manual packing causes undefined behaviour |
| Block size not multiple of 64 | Always align to wavefront size | Partial wavefronts waste SIMD lanes |

---

## 10. Documentation Style for This Project

All experiment code uses **heavily commented educational style**:
- Bilingual comments (Swedish insertions for learning context)
- British English spelling (optimisation, synchronisation, behaviour)
- Every non-obvious line explained
- Hardware behaviour documented alongside code
- CK pattern connections noted where applicable

Comments serve as the primary documentation — no separate README required
for understanding the code itself.

---

*Last updated: February 2026*
*Target: ROCm 7.x, MI300X (gfx942), C++17*
