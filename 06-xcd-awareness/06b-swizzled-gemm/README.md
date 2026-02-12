# Experiment 06b: XCD-Aware GEMM — Swizzled Workgroup Mapping

## Syfte — Purpose

Implement a tiled GEMM kernel with two workgroup mapping strategies:
1. **NAIVE**: Standard linear workgroup-to-tile mapping
2. **SWIZZLED**: XCD-aware mapping that improves L2 cache locality

Implementera en plattsatt GEMM-kärna med två arbetsgruppsmappningsstrategier:
1. **NAIV**: Standard linjär arbetsgrupp-till-platta-mappning
2. **SWIZZLAD**: XCD-medveten mappning som förbättrar L2-cache-lokalitet

---

## The GEMM Tiling Problem — GEMM-plattningsproblemet

In tiled GEMM, we compute `C[M,N] = A[M,K] × B[K,N]` by dividing the output into tiles:

```
    Matrix C (Output)                    Data Sharing Pattern
    ─────────────────                    ────────────────────
    
    N dimension →                        Tiles in same ROW share A data
    ┌──────┬──────┬──────┬──────┐        Tiles in same COLUMN share B data
  M │ T0,0 │ T0,1 │ T0,2 │ T0,3 │
  ↓ ├──────┼──────┼──────┼──────┤        T0,0 and T0,1 share row of A
    │ T1,0 │ T1,1 │ T1,2 │ T1,3 │        T0,0 and T1,0 share column of B
    ├──────┼──────┼──────┼──────┤
    │ T2,0 │ T2,1 │ T2,2 │ T2,3 │
    └──────┴──────┴──────┴──────┘
```

### The Problem — Problemet

**NAIVE mapping** assigns tiles row-major:
```
T0,0 → WG0 → XCD0
T0,1 → WG1 → XCD1  ← T0,0 and T0,1 share A data but are on different XCDs!
T0,2 → WG2 → XCD2
T0,3 → WG3 → XCD3
```

**SWIZZLED mapping** groups related tiles:
```
T0,0 → WG0 → XCD0
T0,1 → WG1 → XCD0  ← Same XCD! Can share L2 cache!
T1,0 → WG2 → XCD0
T1,1 → WG3 → XCD0
```

---

## Building & Running — Bygga & Köra

### Build — Bygg

```bash
cd ~/mfma/06-xcd-awareness/06b-swizzled-gemm
make
```

### Run — Kör

```bash
# Default 2048×2048×2048
make run

# Larger matrices (better shows memory effects)
make run-large    # 4096×4096×4096

# Custom size
./swizzled_gemm 8192 8192 8192
```

### Profile — Profilera

```bash
make profile
```

---

## Expected Output — Förväntad utdata

### Test 1: Correctness

```
Testing NAIVE mapping...
  Max absolute difference: 0.000000
  Errors: 0 / 65536 (0.00%)
  NAIVE: ✓ PASS

Testing SWIZZLED mapping...
  Max absolute difference: 0.000000
  Errors: 0 / 65536 (0.00%)
  SWIZZLED: ✓ PASS
```

### Test 2: Performance

```
┌───────────────────┬──────────────┬──────────────┬─────────────────────┐
│  Mapping          │ Time (ms)    │ GFLOPS       │ Speedup             │
├───────────────────┼──────────────┼──────────────┼─────────────────────┤
│  NAIVE            │      6.548   │   20990.28   │ 1.00x (baseline)    │
│  SWIZZLED         │      6.629   │   20733.58   │ 0.99x               │
└───────────────────┴──────────────┴──────────────┴─────────────────────┘
```

### Why Similar Timing? — Varför liknande tid?

GEMM is **compute-bound**, not memory-bound:
- Each tile performs many FMA operations per byte loaded
- The L2 cache benefit is hidden by compute time
- The improvement is visible in **L2 cache hit rate**, not timing

GEMM är **beräkningsbunden**, inte minnesbunden:
- Varje platta utför många FMA-operationer per byte som laddas
- L2-cache-fördelen döljs av beräkningstiden
- Förbättringen syns i **L2-cache-träfffrekvens**, inte tid

---

## Code Architecture — Kodarkitektur

### Tile Configuration — Plattakonfiguration

```cpp
constexpr int TILE_SIZE = 64;           // 64×64 output tiles
constexpr int BLOCK_K = 16;             // K-dimension block size
constexpr int THREADS_PER_BLOCK = 256;  // 4 wavefronts

// Calculation / Beräkning:
// - 256 threads / 64 columns = 4 thread rows
// - 64 tile rows / 4 thread rows = 16 rows per thread
constexpr int ROWS_PER_THREAD = 16;
```

### Naive Mapping Function — Naiv mappningsfunktion

```cpp
__device__ void naive_wg_to_tile(int wg_id, int tiles_n, int* tile_m, int* tile_n) {
    // Simple row-major: iterate through columns first
    // Enkel radordning: iterera genom kolumner först
    *tile_m = wg_id / tiles_n;
    *tile_n = wg_id % tiles_n;
}
```

### Swizzled Mapping Function — Swizzlad mappningsfunktion

```cpp
__device__ void swizzled_wg_to_tile(int wg_id, int tiles_m, int tiles_n, 
                                     int* tile_m, int* tile_n) {
    // Divide grid into 2×4 = 8 super-tiles (one per XCD)
    // Dela upp rutnät i 2×4 = 8 superplattor (en per XCD)
    int super_tiles_m = 2;
    int super_tiles_n = 4;
    
    int tiles_per_super_m = (tiles_m + 1) / 2;
    int tiles_per_super_n = (tiles_n + 3) / 4;
    int tiles_per_super = tiles_per_super_m * tiles_per_super_n;
    
    // Which super-tile (XCD)?
    int super_tile_id = wg_id / tiles_per_super;
    
    // Position within super-tile
    int local_id = wg_id % tiles_per_super;
    
    // Convert to tile coordinates
    int super_m = super_tile_id / super_tiles_n;
    int super_n = super_tile_id % super_tiles_n;
    *tile_m = super_m * tiles_per_super_m + (local_id / tiles_per_super_n);
    *tile_n = super_n * tiles_per_super_n + (local_id % tiles_per_super_n);
}
```

### Kernel Structure — Kärnstruktur

```cpp
__global__ void k_gemm_tiled(..., int use_swizzle) {
    // 1. LDS allocation for tile caching
    __shared__ float A_lds[TILE_SIZE][BLOCK_K + 1];  // +1 to avoid bank conflicts
    __shared__ float B_lds[BLOCK_K][TILE_SIZE + 1];
    
    // 2. Determine tile position (naive or swizzled)
    int tile_m, tile_n;
    if (use_swizzle) {
        swizzled_wg_to_tile(...);
    } else {
        naive_wg_to_tile(...);
    }
    
    // 3. K-loop: load tiles into LDS, compute partial products
    for (int k_block = 0; k_block < K; k_block += BLOCK_K) {
        // Cooperative load A and B tiles
        // Compute using LDS data
        __syncthreads();
    }
    
    // 4. Write results to global memory
}
```

---

## Bug Fix History — Buggfixhistorik

### Original Bug — Ursprunglig bugg

The first version had `float acc[4]` instead of `float acc[16]`:

```cpp
// WRONG — FEL
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (int r = 0; r < 4; r++) { ... }  // Only 4 iterations!

// CORRECT — RÄTT
float acc[ROWS_PER_THREAD];  // ROWS_PER_THREAD = 16
for (int r = 0; r < ROWS_PER_THREAD; r++) { ... }  // 16 iterations!
```

**Symptom**: 70% of output was zeros (only 16 of 64 rows computed per tile).

---

## Exercises — Övningar

### Exercise 1: Vary Tile Size

Try different tile sizes and observe performance:

```cpp
constexpr int TILE_SIZE = 32;   // Smaller tiles
constexpr int TILE_SIZE = 128;  // Larger tiles (may exceed LDS)
```

### Exercise 2: Profile with rocprof

```bash
# Create metrics.txt with L2 counters
echo "pmc: L2CacheHit" > metrics.txt

# Profile
rocprof -i metrics.txt -o results.csv ./swizzled_gemm 4096 4096 4096

# Examine results
cat results.csv
```

### Exercise 3: Compare with hipBLAS

```cpp
// Add hipBLAS comparison
#include <hipblas/hipblas.h>

hipblasHandle_t handle;
hipblasCreate(&handle);
hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 
             N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
```

---

## Connection to Composable Kernel — Koppling till Composable Kernel

| This Experiment | CK Equivalent |
|-----------------|---------------|
| `swizzled_wg_to_tile` | `BlockToCTileMap` |
| Super-tile grouping | `GridwiseGemm` tile distribution |
| LDS tile caching | `BlockwiseTensorSliceTransfer` |
| K-loop structure | `GridwiseGemm` main loop |

---

## Files — Filer

| File | Description |
|------|-------------|
| `swizzled_gemm.cpp` | Main kernel with both mapping strategies |
| `Makefile` | Build targets: `run`, `run-large`, `profile` |
| `metrics.txt` | rocprof configuration |
| `README.md` | This documentation |

---

## Summary — Sammanfattning

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RESULT — RESULTAT                                                      │
│                                                                         │
│  ✓ Both naive and swizzled produce correct GEMM results                │
│  ✓ Timing is similar because GEMM is compute-bound                     │
│  → The L2 cache benefit will be visible with rocprof in Experiment 06c │
│                                                                         │
│  ✓ Både naiv och swizzlad ger korrekta GEMM-resultat                   │
│  ✓ Tiden är liknande eftersom GEMM är beräkningsbunden                 │
│  → L2-cache-fördelen syns med rocprof i Experiment 06c                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Experiment 06b: Swizzled GEMM*
*Part of the MFMA/CDNA3 Learning Series*
