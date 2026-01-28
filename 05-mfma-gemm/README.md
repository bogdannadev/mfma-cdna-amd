# Experiment 05: MFMA GEMM (Tiled Matrix Multiplication)

## Overview

This experiment implements a practical General Matrix Multiplication (GEMM) kernel using MFMA instructions. GEMM is the computational foundation of modern AI—every transformer layer, convolutional operation, and linear projection ultimately reduces to matrix multiplication.

**Duration:** 2–3 hours  
**Prerequisites:** Experiments 01–04 (HIP basics, wavefront, LDS, MFMA intro)  
**Hardware:** MI300X (gfx942), MI250X (gfx90a), or MI210 (gfx90a)

---

## Learning Objectives

By the end of this experiment, you will:

1. Implement a working MFMA-based GEMM kernel
2. Understand cooperative tile loading from global memory to LDS
3. Apply correct vector types for MFMA intrinsics (reinforcing Exp04)
4. Compare naive vs MFMA performance
5. Know the path towards production-grade GEMM optimisation

---

## GEMM Fundamentals

### The Operation

GEMM computes:

```
D = α × A × B + β × C
```

For simplicity, this experiment uses α = 1, β = 0 or β = 1:

```
D = A × B + C
```

Where:
- **A** — M × K input matrix
- **B** — K × N input matrix  
- **C** — M × N accumulator matrix
- **D** — M × N output matrix

### Why GEMM Matters

| Operation | GEMM Equivalent |
|-----------|-----------------|
| Dense layer | Output = Weights × Input + Bias |
| Attention Q/K/V | Q, K, V = Input × W_q, W_k, W_v |
| Attention scores | Scores = Q × K^T |
| Convolution (im2col) | Output = Filter × Im2col(Input) |
| Batch norm (affine) | Output = γ × Norm(x) + β |

Achieving 80%+ of theoretical FLOPS on GEMM directly translates to inference speed.

---

## Tiled GEMM Strategy

### Why Tiling?

Matrices in real workloads are too large to fit in registers. Tiling breaks the problem into manageable pieces:

```
Full GEMM: D[4096×4096] = A[4096×4096] × B[4096×4096]
                           ↓
Tiled:     For each 16×16 output tile:
             Load A_tile[16×16], B_tile[16×16] from LDS
             D_tile += A_tile × B_tile  (MFMA)
```

### Tiling Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     Full Matrix (M × N)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Block Tile (TB_M × TB_N)            │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │           Wave Tile (TW_M × TW_N)           │     │   │
│  │  │  ┌────────────────────────────────────┐     │     │   │
│  │  │  │     MFMA Tile (16×16 or 32×32)    │     │     │   │
│  │  │  │                                    │     │     │   │
│  │  │  │    One MFMA instruction output    │     │     │   │
│  │  │  └────────────────────────────────────┘     │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

| Level | Size | Managed By |
|-------|------|------------|
| Full Matrix | M × N | Grid of blocks |
| Block Tile | 64–256 | Thread block |
| Wave Tile | 32–64 | Single wavefront |
| MFMA Tile | 16×16 or 32×32 | MFMA instruction |

---

## CRITICAL: Correct Value Packing (Reinforcement)

### The Single Most Important Lesson

From Experiment 04, but worth repeating because this mistake is so common:

```cpp
// ❌ WRONG - Manual bit packing DOES NOT WORK
long a_packed = ((long)(*((int*)&a_hi)) << 32) | (*((int*)&a_lo));

// ✅ CORRECT - Use LLVM ext_vector_type
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
v4f16 a_vec;
a_vec[0] = val0;
a_vec[1] = val1;
a_vec[2] = val2;
a_vec[3] = val3;
```

### Why Bit Packing Fails (Detailed)

When you write:
```cpp
long packed = ((long)hi << 32) | lo;
```

1. **Type punning** — Casting `float*` to `int*` violates strict aliasing rules
2. **Endianness assumptions** — Assumes little-endian byte order
3. **Register allocation** — The compiler sees a 64-bit integer, not 4 × FP16
4. **MFMA encoding** — The instruction expects specific VGPR formatting

The MFMA unit reads data from VGPRs in a specific format. When you pack bits manually, the resulting memory layout doesn't match what MFMA expects, producing garbage results or GPU hangs.

### Correct Vector Type Definitions

```cpp
// For FP32 operations
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v16f32 __attribute__((ext_vector_type(16)));

// For FP16 operations
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
typedef _Float16 v8f16 __attribute__((ext_vector_type(8)));

// NEVER use HIP's float4, half4 - different ABI!
```

---

## Kernel Implementation

### Strategy Overview

1. **Grid mapping** — Each block computes one output tile
2. **Cooperative loading** — All threads in block load A and B tiles to LDS
3. **Synchronisation** — Barrier after loading, before compute
4. **MFMA execution** — Each wavefront executes MFMA with data from LDS
5. **K-loop** — Iterate over K dimension in tile-sized steps
6. **Store results** — Write accumulated output to global memory

### Code Structure

```cpp
__global__ void mfma_gemm_kernel(float *D, const _Float16 *A, const _Float16 *B,
                                  int M, int N, int K) {
    // 1. Thread identification
    int lane_id = threadIdx.x;
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    
    // 2. Shared memory for tiles
    __shared__ _Float16 A_lds[TILE_M * TILE_K];
    __shared__ _Float16 B_lds[TILE_K * TILE_N];
    
    // 3. Accumulator initialisation
    v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // 4. K-dimension loop
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // 4a. Cooperative load A and B tiles to LDS
        load_tile_to_lds(A, A_lds, ...);
        load_tile_to_lds(B, B_lds, ...);
        __syncthreads();
        
        // 4b. Prepare vectors from LDS
        v4f16 a_vec = load_from_lds(A_lds, lane_id);
        v4f16 b_vec = load_from_lds(B_lds, lane_id);
        
        // 4c. Execute MFMA
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, acc, 0, 0, 0);
        
        __syncthreads();
    }
    
    // 5. Store results
    store_tile_to_global(D, acc, ...);
}
```

### Cooperative Tile Loading

All 64 threads collaborate to load the tile:

```cpp
// 256 elements in a 16×16 tile
// 64 threads → each thread loads 4 elements
for (int i = lane_id; i < 256; i += 64) {
    int tile_row = i / 16;
    int tile_col = i % 16;
    
    // Global coordinates
    int global_row = block_row + tile_row;
    int global_col = k_tile + tile_col;
    
    // Bounds check and load
    if (global_row < M && global_col < K) {
        A_lds[i] = A[global_row * K + global_col];
    } else {
        A_lds[i] = (_Float16)0.0f;  // Zero-padding
    }
}
```

### LDS to Vector Loading

The exact LDS-to-lane mapping for MFMA is complex. For educational purposes, a simplified approach:

```cpp
// Each lane loads 4 consecutive FP16 values
int offset = lane_id * 4;
v4f16 a_vec;
a_vec[0] = A_lds[offset % 256];
a_vec[1] = A_lds[(offset + 1) % 256];
a_vec[2] = A_lds[(offset + 2) % 256];
a_vec[3] = A_lds[(offset + 3) % 256];
```

**Note:** Production kernels use optimised layouts matching the AMD Matrix Calculator output. This simplified version is for learning.

---

## The Simpler 4×4×1 MFMA

For verification and understanding, we also implement using `v_mfma_f32_4x4x1_16b_f32`:

### Why 4×4×1?

- **Simpler data layout** — Each lane provides 1 element of A and 1 element of B
- **Easier verification** — Results directly map to standard matrix multiplication
- **Lower register pressure** — 4 output elements vs 16 for larger tiles

### Structure

```cpp
__global__ void mfma_4x4x1_gemm_kernel(float *D, const float *A, const float *B,
                                        int M, int N, int K) {
    int lane_id = threadIdx.x;
    
    // 16 independent 4×4 blocks (64 lanes / 4)
    int block_idx = lane_id / 4;
    int lane_in_block = lane_id % 4;
    
    v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int k = 0; k < K; k++) {
        float a_val = A[...];  // One element per lane
        float b_val = B[...];  // One element per lane
        
        // 4×4 outer product
        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, acc, 0, 0, 0);
    }
    
    // Store 4 elements per lane
    for (int i = 0; i < 4; i++) {
        D[out_row + i, out_col] = acc[i];
    }
}
```

---

## Performance Analysis

### Naive vs MFMA Comparison

| Metric | Naive GEMM | MFMA GEMM | rocBLAS |
|--------|------------|-----------|---------|
| 256×256 Time | ~5 ms | ~0.5 ms | ~0.05 ms |
| GFLOPS | ~10 | ~100 | ~1000 |
| Efficiency | <1% | ~10% | >80% |

### Why Our MFMA Kernel Isn't Fast Yet

1. **Simplified LDS layout** — Not optimised for conflict-free access
2. **No register tiling** — Each wavefront computes only one MFMA tile
3. **No double buffering** — Memory latency not hidden
4. **No software pipelining** — Sequential load → compute → store

---

## Path to Production Performance

### Optimisation 1: Correct Vector Types ✓

Already implemented using `ext_vector_type`.

### Optimisation 2: Register Tiling

Each wavefront computes a larger tile (e.g., 32×32 or 64×64) using multiple MFMA instructions:

```
Wave tile = 4×4 MFMA tiles = 64×64 output
Register file holds all intermediate results
```

### Optimisation 3: Double Buffering

Load next tile while computing current:

```cpp
// Ping-pong buffers
__shared__ _Float16 A_lds[2][TILE_M * TILE_K];
__shared__ _Float16 B_lds[2][TILE_K * TILE_N];

// Load tile 0
load_tile(A_lds[0], B_lds[0], k=0);

for (int k = 0; k < K; k += TILE_K) {
    // Compute with current buffer
    compute_mfma(A_lds[k%2], B_lds[k%2]);
    
    // Simultaneously load next buffer (async)
    if (k + TILE_K < K) {
        load_tile_async(A_lds[(k+1)%2], B_lds[(k+1)%2], k+TILE_K);
    }
}
```

### Optimisation 4: LDS Layout Optimisation

Pad rows to avoid bank conflicts:

```cpp
// Without padding: 32-bank conflicts on column access
__shared__ _Float16 A_lds[16][16];

// With padding: conflict-free
__shared__ _Float16 A_lds[16][16 + 1];
```

### Optimisation 5: Use Production Libraries

For deployment, use highly optimised libraries:

| Library | Use Case |
|---------|----------|
| **rocBLAS** | General BLAS operations |
| **Composable Kernel** | Custom fused operations |
| **hipBLASLt** | Transformer-specific patterns |
| **MIOpen** | Deep learning primitives |

---

## Building and Running

### Build Commands

```bash
# Navigate to experiment directory
cd 05-mfma-gemm

# Build with hipcc
hipcc --offload-arch=gfx942 -O3 -o mfma_gemm mfma_gemm.cpp

# Or use the Makefile
make exp05
```

### Running

```bash
./mfma_gemm
```

Expected output:
- Device information
- 4×4×1 GEMM test with verification
- Naive GPU GEMM benchmark
- Optimisation guide

### Generating Assembly

```bash
hipcc --offload-arch=gfx942 -O3 -S -o mfma_gemm.s mfma_gemm.cpp
grep -i "mfma" mfma_gemm.s
```

---

## Exercises

### Exercise 1: Verify Correctness

Modify the test to use random values instead of all 1s:

```cpp
for (int i = 0; i < M * K; i++)
    h_A[i] = (float)(rand() % 10) / 10.0f;
```

Does the output still match the CPU reference?

### Exercise 2: Tile Size Exploration

Modify `TILE_SIZE` from 16 to 32:

1. How does register usage change?
2. What happens to occupancy?
3. Is performance better or worse?

### Exercise 3: Compare with rocBLAS

Add rocBLAS comparison:

```cpp
#include <rocblas/rocblas.h>

rocblas_handle handle;
rocblas_create_handle(&handle);
rocblas_sgemm(handle, ...);
```

How much faster is rocBLAS?

### Exercise 4: Assembly Analysis

Find in the assembly:
1. How many `v_mfma_*` instructions are generated?
2. Where are `ds_read` and `ds_write` (LDS operations)?
3. Are there `v_accvgpr_read` instructions (AGPR to VGPR moves)?

---

## Connection to Composable Kernel

This experiment's patterns map to CK components:

| This Experiment | CK Component |
|-----------------|--------------|
| Block tile loop | `BlockwiseGemmBlockLoop` |
| LDS tile loading | `BlockwiseTensorSliceTransfer` |
| MFMA execution | `WavewiseGemmMfma` |
| Output store | `ThreadwiseTensorSliceTransfer` |

Understanding these raw operations prepares you for CK's template-based abstractions.

---

## Course Summary

Congratulations! You've completed the MFMA course core experiments.

### What You've Learned

| Experiment | Key Takeaways |
|------------|---------------|
| 01 Hello HIP | Kernel launch, memory transfers, thread indexing |
| 02 Wavefront | 64-thread SIMD, lane operations, divergence costs |
| 03 LDS Memory | Shared memory, bank conflicts, synchronisation |
| 04 MFMA Intro | Matrix cores, register distribution, **correct vector types** |
| 05 MFMA GEMM | Tiled GEMM, cooperative loading, optimisation path |

### The Critical Lesson

**MFMA intrinsics require LLVM ext_vector_type, NOT manual bit packing!**

```cpp
// ✅ ALWAYS use this
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
```

### Continue Your Journey

1. **rocBLAS/hipBLAS** — Production GEMM for inference
2. **Composable Kernel** — Custom fused operations
3. **rocprof** — Profile to find bottlenecks
4. **AMD Developer Cloud** — Extended access for contributions

---

## References

- [AMD Instinct MI300 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [rocBLAS Source](https://github.com/ROCm/rocBLAS)
- [Composable Kernel](https://github.com/ROCm/composable_kernel)
- [AMD Matrix Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)