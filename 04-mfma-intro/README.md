# Experiment 04: MFMA (Matrix Fused Multiply-Add) Introduction

## Overview

This experiment introduces AMD's Matrix Core instructions—the computational heart of modern AI acceleration on CDNA architectures. You'll learn how a single wavefront (64 threads) collaboratively executes matrix operations through specialised hardware.

**Duration:** 2–3 hours  
**Prerequisites:** Experiments 01–03 (HIP basics, wavefront execution, LDS memory)  
**Hardware:** MI300X (gfx942), MI250X (gfx90a), or MI210 (gfx90a)

---

## Learning Objectives

By the end of this experiment, you will:

1. Understand the MFMA operation: D = A × B + C
2. Know how matrix elements distribute across 64 wavefront lanes
3. Recognise the difference between VGPRs (inputs) and AGPRs (accumulators)
4. Use the correct LLVM vector types for MFMA intrinsics
5. Avoid the common "manual bit packing" anti-pattern

---

## Theoretical Background

### What is MFMA?

**Matrix Fused Multiply-Add** (MFMA) instructions perform matrix-matrix multiplication with accumulation in a single instruction:

```
D = A × B + C
```

Where:
- **A** — Input matrix (stored in VGPRs)
- **B** — Input matrix (stored in VGPRs)
- **C** — Accumulator input (stored in AGPRs)
- **D** — Result matrix (written to AGPRs, often in-place with C)

From the MI300 ISA (Section 7.1):
> "The matrix fused-multiply-add (MFMA) instructions use the matrix core to perform one or more matrix multiplications."
> "The matrix core unit... has the 4×1 by 1×4 outer product as its fundamental computational primitive."

### MFMA Naming Convention

MFMA instructions follow a predictable naming pattern:

```
V_MFMA_[OutputType]_[M]X[N]X[K]_[InputType]
```

| Component | Meaning |
|-----------|---------|
| OutputType | Data type of output elements (F32, F64) |
| M×N | Output matrix dimensions |
| K | Reduction dimension (inner product length) |
| InputType | Data type of input elements (F16, BF16, F32) |

**Example:** `v_mfma_f32_16x16x16_f16`
- 16×16 output matrix with FP32 elements
- 16-element reduction dimension (K=16)
- FP16 input elements
- Completes in 16 cycles

### Common MFMA Instructions for LLM Inference

| Instruction | Output | Cycles | Use Case |
|-------------|--------|--------|----------|
| `v_mfma_f32_16x16x16_f16` | 16×16 | 16 | LLM FP16 inference |
| `v_mfma_f32_32x32x8_f16` | 32×32 | 32 | Large tile FP16 |
| `v_mfma_f32_16x16x16_bf16` | 16×16 | 16 | BF16 training/inference |
| `v_mfma_f32_4x4x1_16b_f32` | 4×4×16 | 8 | FP32 demonstration |
| `v_mfma_f64_16x16x4_f64` | 16×16 | 32 | FP64 HPC workloads |

The 16×16×16 FP16 variant is the "sweet spot" for transformer inference—it balances tile size, register pressure, and throughput.

---

## Register Architecture

### Register Types

AMD GPUs use three primary register types for MFMA operations:

| Register | Full Name | Purpose | MFMA Role |
|----------|-----------|---------|-----------|
| VGPR | Vector General Purpose Register | Per-lane data | A, B inputs |
| AGPR | Accumulation GPR | Matrix accumulation | C, D matrices |
| SGPR | Scalar GPR | Uniform values | Loop counters, addresses |

### Data Distribution Across Lanes

For `v_mfma_f32_16x16x16_f16`:

```
Total output elements: 16 × 16 = 256
Wavefront lanes:       64
Elements per lane:     256 / 64 = 4
```

Each lane computes and stores **4 elements** of the output matrix. The exact mapping of which lane holds which elements is complex—use AMD's Matrix Calculator to visualise it:

```bash
./matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_f16 \
    --detail-instruction
```

### Register Usage Summary

```
┌────────────┬────────────────────────────────────────────┐
│ Matrix     │ Registers                                  │
├────────────┼────────────────────────────────────────────┤
│ A (input)  │ 4 VGPRs (64-bit, holds 4×FP16 per lane)    │
│ B (input)  │ 4 VGPRs (64-bit, holds 4×FP16 per lane)    │
│ C (accum)  │ 4 AGPRs (holds 4×FP32 per lane)            │
│ D (output) │ 4 AGPRs (same as C, in-place)              │
└────────────┴────────────────────────────────────────────┘
```

### Data Movement Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Global Memory (HBM)                                    │
│         ↓ (global_load)                                 │
│  LDS (Local Data Share) - 64KB                          │
│         ↓ (ds_read)                                     │
│  VGPRs (A, B matrices)                                  │
│         ↓ (v_mfma instruction)                          │
│  Matrix Core → AGPRs (C/D matrices)                     │
│         ↓ (v_accvgpr_read + global_store)               │
│  Global Memory (HBM)                                    │
└─────────────────────────────────────────────────────────┘
```

---

## CRITICAL: Value Packing for MFMA Intrinsics

### The Anti-Pattern (DO NOT USE)

A common mistake when first working with MFMA is attempting to manually pack FP16 values into integer registers using bit manipulation:

```cpp
// ❌ WRONG - Manual bit packing
_Float16 a_lo = ...; 
_Float16 a_hi = ...;

// This DOES NOT work correctly with MFMA intrinsics!
long a_packed = ((long)(*((int*)&a_hi)) << 32) | (*((int*)&a_lo));
```

**Why this fails:**
1. **ABI mismatch** — LLVM's ext_vector_type has specific alignment and register allocation rules
2. **Undefined behaviour** — Type punning via pointer casts violates strict aliasing
3. **Wrong register mapping** — The packed long doesn't map to the VGPR layout MFMA expects
4. **Compiler cannot optimise** — The compiler loses semantic information about the data

### The Correct Approach (USE THIS)

MFMA intrinsics require LLVM's `ext_vector_type` attribute:

```cpp
// ✅ CORRECT - LLVM ext_vector_type
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));

// Create the vector properly
v4f16 a_vec;
a_vec[0] = value0;  // Element 0
a_vec[1] = value1;  // Element 1
a_vec[2] = value2;  // Element 2
a_vec[3] = value3;  // Element 3

// Pass to MFMA intrinsic
v4f32 result = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, c_vec, 0, 0, 0);
```

### Why ext_vector_type Works

1. **Compiler-aware** — The compiler understands the vector's semantics and can optimise
2. **Correct ABI** — LLVM generates proper VGPR allocation matching MFMA requirements
3. **Type-safe** — No undefined behaviour from pointer aliasing
4. **Portable** — Works across compiler versions and optimisation levels

### Common Vector Type Definitions

```cpp
// FP32 vectors
typedef float v4f32 __attribute__((ext_vector_type(4)));   // 4 floats
typedef float v16f32 __attribute__((ext_vector_type(16))); // 16 floats

// FP16 vectors  
typedef _Float16 v4f16 __attribute__((ext_vector_type(4))); // 4 halfs
typedef _Float16 v8f16 __attribute__((ext_vector_type(8))); // 8 halfs

// DO NOT use HIP's float4 or half4 - they have different ABI!
```

### Visual Comparison

```
WRONG (bit packing):
┌─────────────────────────────────────────────────────────┐
│ long packed = (hi << 32) | lo                           │
│                                                         │
│ Memory:  [????][????][lo_bits][hi_bits]                 │
│ VGPR:    Unpredictable register allocation              │
│ Result:  Garbage output from MFMA                       │
└─────────────────────────────────────────────────────────┘

CORRECT (ext_vector_type):
┌─────────────────────────────────────────────────────────┐
│ v4f16 vec = {v0, v1, v2, v3}                            │
│                                                         │
│ Memory:  [v0][v1][v2][v3] (packed FP16)                 │
│ VGPR:    Proper 64-bit VGPR with 4 × FP16               │
│ Result:  Correct MFMA computation                       │
└─────────────────────────────────────────────────────────┘
```

---

## MFMA Intrinsic API

### Function Signature

```cpp
v4f32 __builtin_amdgcn_mfma_f32_16x16x16f16(
    v4f16 A,     // Input matrix A (4 × FP16 per lane)
    v4f16 B,     // Input matrix B (4 × FP16 per lane)
    v4f32 C,     // Accumulator input (4 × FP32 per lane)
    int cbsz,    // Control broadcast size (usually 0)
    int abid,    // A-matrix broadcast ID (usually 0)
    int blgp     // B-matrix lane group pattern (usually 0)
);
```

### Control Parameters

For most use cases, set all control parameters to 0:

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| cbsz | Broadcast control for sub-tile operations | 0 |
| abid | A-matrix broadcast identifier | 0 |
| blgp | B-matrix lane group pattern | 0 |

Advanced uses of these parameters enable sub-tile broadcasting and complex data reuse patterns—see the ISA manual Section 7.1 for details.

### Available Intrinsics (gfx942)

```cpp
// FP16 → FP32
__builtin_amdgcn_mfma_f32_16x16x16f16(v4f16, v4f16, v4f32, int, int, int)
__builtin_amdgcn_mfma_f32_32x32x8f16(v4f16, v4f16, v16f32, int, int, int)

// BF16 → FP32  
__builtin_amdgcn_mfma_f32_16x16x16bf16(v4bf16, v4bf16, v4f32, int, int, int)
__builtin_amdgcn_mfma_f32_32x32x8bf16(v4bf16, v4bf16, v16f32, int, int, int)

// FP32 → FP32
__builtin_amdgcn_mfma_f32_4x4x1f32(float, float, v4f32, int, int, int)
__builtin_amdgcn_mfma_f32_16x16x4f32(v4f32, v4f32, v16f32, int, int, int)

// FP64 → FP64
__builtin_amdgcn_mfma_f64_16x16x4f64(double, double, v4f64, int, int, int)
```

---

## Building and Running

### Build Commands

```bash
# Navigate to experiment directory
cd 04-mfma-intro

# Build with hipcc
hipcc --offload-arch=gfx942 -O3 -o mfma_intro mfma_intro.cpp

# Or use the Makefile from repository root
make exp04
```

### Running

```bash
./mfma_intro
```

Expected output includes:
- Device detection and MFMA support verification
- MFMA instruction overview
- Register usage explanation
- Test execution with a 4×4×1 FP32 demonstration

### Examining Assembly

Understanding the generated assembly is crucial for optimisation:

```bash
# Generate assembly
hipcc --offload-arch=gfx942 -O3 -S -o mfma_intro.s mfma_intro.cpp

# Or save all intermediate files
hipcc --offload-arch=gfx942 -O3 --save-temps -o mfma_intro mfma_intro.cpp
```

Look for MFMA instructions in the output:

```asm
v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[0:3]
```

---

## Exercises

### Exercise 1: Visualise Data Layouts

Use AMD's Matrix Calculator to understand the exact lane-to-element mapping:

```bash
./matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_f16 \
    --matrix-layout
```

Questions to answer:
1. Which lanes hold elements from row 0 of the output?
2. Which lanes hold elements from column 0?
3. How does the layout change for 32×32×8?

### Exercise 2: Assembly Analysis

Generate and examine the assembly:

```bash
hipcc --offload-arch=gfx942 -O3 -S -o mfma_intro.s mfma_intro.cpp
grep -i "mfma" mfma_intro.s
```

Questions:
1. How many MFMA instructions are generated?
2. What registers are used for inputs (v) vs accumulators (a)?
3. Are there any data movement instructions before the MFMA?

### Exercise 3: Modify for Different Tile Sizes

Modify the kernel to use `v_mfma_f32_32x32x8_f16`:
1. What changes in register allocation?
2. How does the output layout differ?
3. What's the performance difference for larger matrices?

---

## Connection to Composable Kernel

The patterns in this experiment directly map to CK abstractions:

| This Experiment | CK Equivalent |
|-----------------|---------------|
| v4f16 vector type | `vector_type<half_t, 4>` |
| Lane-to-element mapping | `WaveGemmMfmaF16F32` |
| AGPR accumulator | Accumulator fragments |
| LDS loading | `BlockwiseTensorSliceTransfer` |

Understanding the raw MFMA intrinsics prepares you for CK's higher-level templated approach.

---

## Common Mistakes

### 1. Using HIP vector types

```cpp
// ❌ WRONG
float4 my_vec;  // HIP type - different ABI

// ✅ CORRECT  
v4f32 my_vec;   // LLVM ext_vector_type
```

### 2. Forgetting accumulator initialisation

```cpp
// ❌ WRONG - uninitialised accumulator
v4f32 acc;
acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);

// ✅ CORRECT - zero-initialised
v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};
acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
```

### 3. Wrong thread count

```cpp
// ❌ WRONG - MFMA needs exactly 64 threads
kernel<<<1, 32>>>();  // Only 32 threads!

// ✅ CORRECT - Full wavefront
kernel<<<1, 64>>>();  // 64 threads = 1 wavefront
```

### 4. Missing architecture flag

```bash
# ❌ WRONG - default architecture may not support MFMA
hipcc -o mfma_intro mfma_intro.cpp

# ✅ CORRECT - specify CDNA architecture
hipcc --offload-arch=gfx942 -o mfma_intro mfma_intro.cpp
```

---

## Key Takeaways

1. **MFMA = Matrix Fused Multiply-Add** — D = A × B + C in hardware
2. **One wavefront (64 threads) executes one MFMA** — All lanes participate
3. **Matrix elements distribute across lanes** — 4 elements per lane for 16×16
4. **Use ext_vector_type, NOT bit packing** — Critical for correctness
5. **VGPRs hold inputs (A, B), AGPRs hold accumulators (C, D)**
6. **16×16×16 FP16 is the sweet spot** for LLM inference

---

## Next Steps

Proceed to **Experiment 05: MFMA GEMM** to build a complete tiled matrix multiplication kernel using these fundamentals.

---

## References

- [AMD Instinct MI300 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — Section 7.1
- [LLVM AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html) — Intrinsic documentation
- [AMD Matrix Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) — Layout visualisation tool