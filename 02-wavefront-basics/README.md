# Experiment 02: Wavefront Basics

This experiment reveals how threads actually execute on AMD GPUs. Understanding wavefront execution is essential—every optimisation technique in later experiments builds on these concepts.

---

## The SIMD Execution Model

AMD GPUs execute threads in groups of 64 called **wavefronts**. All 64 threads execute the *same instruction* at the *same time*, but on *different data*.

```
                    Single Instruction
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    SIMD Unit                            │
    │                                                         │
    │  Lane 0   Lane 1   Lane 2   ...   Lane 62   Lane 63    │
    │    │        │        │               │         │        │
    │    ▼        ▼        ▼               ▼         ▼        │
    │  ┌───┐    ┌───┐    ┌───┐           ┌───┐    ┌───┐      │
    │  │ + │    │ + │    │ + │    ...    │ + │    │ + │      │
    │  └───┘    └───┘    └───┘           └───┘    └───┘      │
    │    │        │        │               │         │        │
    │    ▼        ▼        ▼               ▼         ▼        │
    │  Data 0  Data 1   Data 2  ...    Data 62   Data 63     │
    └─────────────────────────────────────────────────────────┘
```

This is called **SIMT** (Single Instruction, Multiple Threads) or **SIMD** (Single Instruction, Multiple Data).

### Why 64 Threads?

AMD chose 64 threads per wavefront for several reasons:

1. **Register efficiency**: 64 is a power of 2 that maps well to hardware register banks.

2. **Memory coalescing**: 64 threads × 4 bytes = 256 bytes, which aligns with cache line sizes.

3. **Instruction throughput**: The SIMD unit can perform 64 operations per cycle.

NVIDIA uses 32-thread warps. Neither is "better"—they're different engineering trade-offs. Larger wavefronts amortise instruction fetch/decode overhead but may suffer more from divergence.

---

## Lane Identity: Your Position in the Wavefront

Each thread in a wavefront has a **lane ID** from 0 to 63. This identity is crucial for understanding how data maps to computation.

```cpp
int lane_id = threadIdx.x % 64;
```

### Why Lane ID Matters

In MFMA (Matrix Fused Multiply-Add) operations, each lane computes a specific element of the output matrix. The lane ID determines exactly which elements that thread is responsible for.

```
32×32 MFMA Output Matrix

Lane 0  computes: [0,0], [0,1], [0,2], ...
Lane 1  computes: [1,0], [1,1], [1,2], ...
Lane 2  computes: [2,0], [2,1], [2,2], ...
...
Lane 63 computes: [63,0], [63,1], [63,2], ...
```

Understanding this mapping is essential for Experiments 04 and 05.

---

## Register Types: SGPRs and VGPRs

AMD GPUs have two types of general-purpose registers:

### Scalar GPRs (SGPRs)
- **One value shared** across the entire wavefront
- Used for values that are the same for all threads
- Examples: `blockIdx.x`, loop counters, pointers to global memory
- Limited quantity (~100 per wavefront)

### Vector GPRs (VGPRs)
- **One value per lane** (64 values per wavefront)
- Used for per-thread data
- Examples: `threadIdx.x`, array elements, computation results
- More plentiful (~256 per wavefront on MI300X)

```
SGPR (Scalar):                VGPR (Vector):
┌─────────────┐               ┌───┬───┬───┬───┬───┬───┐
│   blockIdx  │               │ 0 │ 1 │ 2 │...│62 │63 │  ← lane
│     = 5     │               ├───┼───┼───┼───┼───┼───┤
│ (all lanes) │               │v0 │v1 │v2 │...│v62│v63│  ← values
└─────────────┘               └───┴───┴───┴───┴───┴───┘
```

### The Compiler's Job

The compiler automatically decides what goes in SGPRs vs VGPRs:

```cpp
__global__ void example(float* data, int n) {
    // 'n' is same for all threads → likely SGPR
    // 'data' pointer is same for all threads → SGPR
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 'tid' is different per thread → VGPR
    
    float val = data[tid];
    // 'val' is different per thread → VGPR
}
```

Understanding this helps you predict performance: SGPR operations are "free" (shared), VGPR operations happen 64 times in parallel.

---

## The EXEC Mask: Controlling Which Lanes Execute

Not all 64 lanes always do useful work. The **EXEC mask** is a 64-bit value where each bit controls whether that lane executes instructions.

```
EXEC = 0xFFFFFFFFFFFFFFFF  (all lanes active)
       
       Lane: 0         16        32        48        63
             |         |         |         |         |
             ▼         ▼         ▼         ▼         ▼
             ██████████████████████████████████████████
             (all 64 bits = 1, all lanes execute)
       
EXEC = 0x000000000000FFFF  (only lanes 0-15 active)
       
       Lane: 0         16        32        48        63
             |         |         |         |         |
             ▼         ▼         ▼         ▼         ▼
             ████████████________________________________
             (bits 0-15 = 1, lanes 16-63 masked off)
```

### When EXEC Changes

1. **Bounds checking**: If some threads are beyond array bounds, their lanes are masked off.

2. **Conditional branches**: When threads take different paths, lanes are selectively disabled.

3. **Predication**: Some instructions can conditionally skip based on per-lane conditions.

---

## Branch Divergence: The Performance Killer

When threads in a wavefront take different paths through `if/else`, something expensive happens:

```cpp
if (lane_id < 32) {
    path_A();  // Executed by lanes 0-31
} else {
    path_B();  // Executed by lanes 32-63
}
```

The hardware cannot execute both paths simultaneously. Instead:

```
Step 1: EXEC = 0x00000000FFFFFFFF (lanes 0-31 active)
        Execute path_A()
        
Step 2: EXEC = 0xFFFFFFFF00000000 (lanes 32-63 active)
        Execute path_B()
        
Total time: path_A + path_B (serialised!)
```

### Divergence Example

```cpp
// WORST CASE: 64-way divergence
switch (lane_id % 4) {
    case 0: a(); break;
    case 1: b(); break;
    case 2: c(); break;
    case 3: d(); break;
}
// Executes a(), then b(), then c(), then d() = 4× slowdown
```

```cpp
// BETTER: Uniform within wavefront
switch (blockIdx.x % 4) {  // Same for all threads in block
    case 0: a(); break;
    case 1: b(); break;
    case 2: c(); break;
    case 3: d(); break;
}
// All lanes take same path = no divergence
```

### When Divergence Is Acceptable

Sometimes divergence is unavoidable. The question is: how much work happens in each branch?

```cpp
// Acceptable: simple assignment diverges
output[tid] = (condition) ? value_a : value_b;  // Usually predicated

// Problematic: complex computation diverges
if (condition) {
    expensive_calculation();  // This serialises
}
```

---

## Cross-Lane Communication: Shuffle Operations

Sometimes threads need to share data within a wavefront. AMD provides intrinsics for this:

### `__shfl(value, source_lane)`
Read a value from a specific lane.

```cpp
float x = __shfl(my_value, 0);  // Everyone gets lane 0's value
```

### `__shfl_xor(value, mask)`
Exchange with lane at `my_lane ^ mask`.

```cpp
// Butterfly reduction pattern
val += __shfl_xor(val, 1);   // Exchange with neighbour
val += __shfl_xor(val, 2);   // Exchange across pairs
val += __shfl_xor(val, 4);   // Exchange across quads
...
val += __shfl_xor(val, 32);  // Final exchange
// Lane 0 now has sum of all 64 values
```

### `__shfl_down(value, delta)`
Read from `my_lane + delta`.

```cpp
float neighbor = __shfl_down(my_value, 1);  // Get next lane's value
```

### Why Shuffles Are Fast

Shuffle operations happen within the register file—no memory access required. They complete in a few cycles, making them ideal for:
- Reductions (sum, min, max)
- Prefix sums (scan)
- Broadcast values
- Transpose operations

---

## Wavefront-Level Thinking

When writing GPU code, think at the wavefront level, not the thread level.

### Memory Access

```cpp
// Good: Wavefront accesses contiguous memory
data[tid]        // Lanes 0-63 access addresses 0-63 = coalesced

// Bad: Strided access
data[tid * 32]   // Lanes 0-63 access addresses 0, 32, 64... = scattered
```

### Computation

```cpp
// Good: Uniform control flow
for (int i = 0; i < N; i++) {  // N is constant
    process(data[tid + i * stride]);
}

// Bad: Variable control flow
for (int i = 0; i < per_thread_count[tid]; i++) {  // Different per thread
    process(...);  // Divergent termination
}
```

---

## Terminology Translation

If you're coming from NVIDIA CUDA, here's the mapping:

| AMD Term | NVIDIA Term | Meaning |
|----------|-------------|---------|
| Wavefront | Warp | 64/32 threads executing together |
| Lane | Lane | Thread position within wavefront/warp |
| Compute Unit (CU) | Streaming Multiprocessor (SM) | Hardware execution unit |
| LDS | Shared Memory | Fast on-chip memory |
| VGPR | Register | Per-thread register |
| SGPR | (Uniform register) | Wavefront-shared register |

---

## Exercises

1. **Visualise wavefronts**: Launch 256 threads and print `threadIdx.x`, `threadIdx.x / 64` (wavefront), and `threadIdx.x % 64` (lane). Verify you get 4 wavefronts.

2. **Measure divergence**: Create two kernels—one with uniform branching, one with per-lane branching. Time them with `hipEvent` and compare.

3. **Implement reduction**: Sum an array using only shuffle operations (no shared memory). Compare performance to an LDS-based reduction.

4. **EXEC mask exploration**: Use `__ballot()` to observe which lanes are active at different points in a kernel with branches.

---

## Connection to MFMA

In Experiment 04, you'll see MFMA instructions like:

```cpp
v_mfma_f32_32x32x8_f16(...)
```

These operate on the entire wavefront simultaneously. Each lane contributes input elements and receives output elements based on its lane ID. The mapping is complex and architecture-specific—understanding it requires the foundation you're building now.

---

## What's Next

Experiment 03 introduces LDS (Local Data Share)—fast shared memory that enables cooperation between threads in a block. Combined with wavefront understanding, LDS is the key to efficient matrix tiling.

---

## Quick Reference

| Concept | Description |
|---------|-------------|
| Wavefront | 64 threads executing in SIMD |
| Lane ID | Thread's position (0-63) within wavefront |
| SGPR | Scalar register, one value per wavefront |
| VGPR | Vector register, one value per lane |
| EXEC mask | 64-bit mask controlling active lanes |
| Divergence | When lanes take different branch paths |
| Shuffle | Cross-lane data exchange within wavefront |

| Intrinsic | Purpose |
|-----------|---------|
| `__shfl(val, lane)` | Read from specific lane |
| `__shfl_xor(val, mask)` | Exchange with lane XOR'd with mask |
| `__shfl_down(val, delta)` | Read from lane + delta |
| `__shfl_up(val, delta)` | Read from lane - delta |
| `__ballot(predicate)` | Create mask of lanes where predicate is true |
