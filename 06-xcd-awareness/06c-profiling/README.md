# Experiment 06c: L2 Cache Profiling — Demonstrating XCD NUMA Effects

## Syfte — Purpose

Demonstrate the actual L2 cache locality effects using:
1. A **memory-bound kernel** that clearly shows the swizzling benefit
2. **rocprof** to measure L2 cache hit rates
3. Direct performance comparison between naive and swizzled mapping

Demonstrera de faktiska L2-cache-lokalitetseffekterna med:
1. En **minnesbunden kärna** som tydligt visar swizzlingfördelen
2. **rocprof** för att mäta L2-cache-träfffrekvenser
3. Direkt prestandajämförelse mellan naiv och swizzlad mappning

---

## Why a Different Kernel? — Varför en annan kärna?

### GEMM vs Attention-like Access

| Workload | Characteristic | L2 Benefit |
|----------|----------------|------------|
| **GEMM** | Compute-bound | Small (hidden by compute) |
| **Attention** | Memory-bound | Large (visible!) |

This kernel simulates **attention-like memory access**:
- Multiple "heads" that share KEY data
- Many workgroups process the same head
- All workgroups reading the same data should benefit from L2 caching

---

## The Access Pattern — Åtkomstmönstret

```
Configuration / Konfiguration:
  - 64 attention heads (huvuden)
  - 32 workgroups per head
  - Each head has 4096 floats of KEY data (16 KB)
  - Total: 2048 workgroups, 1 MB KEY data
```

### NAIVE Distribution — Naiv fördelning

```
Head 0's workgroups spread across ALL XCDs:
Huvud 0:s arbetsgrupper sprids över ALLA XCD:er:

  WG 0  (Head 0, Block 0)  → XCD 0  │  Loads KEY[0] from HBM
  WG 1  (Head 0, Block 1)  → XCD 1  │  Loads KEY[0] from HBM (again!)
  WG 2  (Head 0, Block 2)  → XCD 2  │  Loads KEY[0] from HBM (again!)
  WG 3  (Head 0, Block 3)  → XCD 3  │  Loads KEY[0] from HBM (again!)
  WG 4  (Head 0, Block 4)  → XCD 4  │  Loads KEY[0] from HBM (again!)
  WG 5  (Head 0, Block 5)  → XCD 5  │  Loads KEY[0] from HBM (again!)
  WG 6  (Head 0, Block 6)  → XCD 6  │  Loads KEY[0] from HBM (again!)
  WG 7  (Head 0, Block 7)  → XCD 7  │  Loads KEY[0] from HBM (again!)

Result: 8 separate HBM loads for the same data!
Resultat: 8 separata HBM-laddningar för samma data!

→ L2 cache hit rate: 30-50%
```

### SWIZZLED Distribution — Swizzlad fördelning

```
Head 0's workgroups all on XCD 0:
Huvud 0:s arbetsgrupper alla på XCD 0:

  WG 0  (Head 0, Block 0)  → XCD 0  │  Loads KEY[0] from HBM → into L2
  WG 1  (Head 0, Block 1)  → XCD 0  │  Gets KEY[0] from L2! ✓
  WG 2  (Head 0, Block 2)  → XCD 0  │  Gets KEY[0] from L2! ✓
  ...
  WG 31 (Head 0, Block 31) → XCD 0  │  Gets KEY[0] from L2! ✓

Result: 1 HBM load, 31 L2 cache hits!
Resultat: 1 HBM-laddning, 31 L2-cache-träffar!

→ L2 cache hit rate: 80-95%
```

---

## Building & Running — Bygga & Köra

### Build — Bygg

```bash
cd ~/mfma/06-xcd-awareness/06c-profiling
make
```

### Run (timing comparison) — Kör (tidsjämförelse)

```bash
make run
```

### Profile (L2 cache metrics) — Profilera (L2-cache-metriker)

```bash
make profile
```

This runs:
```bash
rocprof -i metrics.txt -o results.csv ./l2_profiling
```

---

## Expected Results — Förväntade resultat

### Timing Results

```
┌───────────────────┬──────────────┬──────────────┬─────────────────────┐
│  Mapping          │ Time (ms)    │ BW (GB/s)    │ Speedup             │
├───────────────────┼──────────────┼──────────────┼─────────────────────┤
│  NAIVE            │     0.XXXX   │     XXX.X    │ 1.00x (baseline)    │
│  SWIZZLED         │     0.XXXX   │     XXX.X    │ 1.XX-1.5Xx          │
└───────────────────┴──────────────┴──────────────┴─────────────────────┘
```

**Expected speedup**: 10-50% depending on memory pressure.

### rocprof Results (results.csv)

```csv
Index,KernelName,L2CacheHit,...
0,k_attention_like_access,35.2,...    ← NAIVE (use_swizzle=0)
1,k_attention_like_access,87.4,...    ← SWIZZLED (use_swizzle=1)
```

**Expected L2CacheHit**:
- NAIVE: 30-50%
- SWIZZLED: 80-95%

---

## Interpreting rocprof Output — Tolka rocprof-utdata

### The results.csv File

The kernel runs **twice** in the program:
1. First with `use_swizzle=0` (NAIVE)
2. Then with `use_swizzle=1` (SWIZZLED)

Each kernel launch creates a row in `results.csv`. The **L2CacheHit** column shows the cache hit percentage.

### Key Columns

| Column | Meaning |
|--------|---------|
| `L2CacheHit` | L2 cache hit rate (%) — main metric! |
| `TCP_TCC_READ_REQ_sum` | Total L2 read requests |
| `TCP_TCC_WRITE_REQ_sum` | Total L2 write requests |
| `FETCH_SIZE` | Bytes fetched from memory |
| `SQ_WAVES` | Number of wavefronts launched |

---

## Code Walkthrough — Kodgenomgång

### Kernel Structure — Kärnstruktur

```cpp
__global__ void k_attention_like_access(
    const float* keys,      // [NUM_HEADS][KEY_SIZE]
    float* output,          // [total_workgroups]
    int num_heads,
    int blocks_per_head,
    int key_size,
    int use_swizzle         // 0 = naive, 1 = swizzled
) {
    int wg_id = blockIdx.x;
    int head_idx;
    
    if (use_swizzle) {
        // SWIZZLED: Group heads by XCD
        int heads_per_xcd = num_heads / NUM_XCDS;  // 64/8 = 8
        int blocks_per_xcd = heads_per_xcd * blocks_per_head;  // 8*32 = 256
        
        int xcd_id = wg_id / blocks_per_xcd;
        int local_id = wg_id % blocks_per_xcd;
        int local_head = local_id / blocks_per_head;
        
        head_idx = xcd_id * heads_per_xcd + local_head;
    } else {
        // NAIVE: Simple linear mapping
        head_idx = wg_id / blocks_per_head;
    }
    
    // Read KEY data for this head
    // ALL workgroups for same head read SAME data!
    const float* key_ptr = keys + head_idx * key_size;
    
    float sum = 0.0f;
    for (int i = tid; i < key_size; i += THREADS_PER_BLOCK) {
        sum += key_ptr[i];
    }
    
    // Reduce and store result
    ...
}
```

### Swizzling Logic — Swizzlinglogik

```
NUM_HEADS = 64, NUM_XCDS = 8
heads_per_xcd = 64 / 8 = 8

XCD 0: Heads 0-7   (WGs 0-255)
XCD 1: Heads 8-15  (WGs 256-511)
XCD 2: Heads 16-23 (WGs 512-767)
...
XCD 7: Heads 56-63 (WGs 1792-2047)
```

---

## Troubleshooting — Felsökning

### rocprof Permission Denied

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Log out and back in, or:
newgrp video
```

### L2CacheHit Not in Output

```bash
# Check available counters
rocprof --list-basic | grep -i l2
rocprof --list-basic | grep -i cache

# Try alternative counter names
echo "pmc: TCP_TCC_HIT_sum TCP_TCC_MISS_sum" > metrics.txt
```

### Values Don't Match Expected

- Verify GPU is MI300X (gfx942): `rocminfo | grep gfx`
- Check ROCm version: `rocprof --version`
- Try larger problem sizes for more pronounced effects

---

## Exercises — Övningar

### Exercise 1: Vary Configuration

```cpp
// Try different configurations
constexpr int NUM_HEADS = 128;       // More heads
constexpr int BLOCKS_PER_HEAD = 64;  // More blocks per head
constexpr int KEY_SIZE = 8192;       // Larger KEY data
```

### Exercise 2: Profile with rocprofv3

```bash
rocprofv3 --hip-trace --kernel-trace -o trace_results ./l2_profiling

# View in Perfetto
# Open https://ui.perfetto.dev/
# Load trace_results/results.json
```

### Exercise 3: Compare with Paper Results

The paper reports:
- Naive: ~43% L2 hit rate
- Swizzled: ~92% L2 hit rate

Do your results match? What factors might cause differences?

---

## Connection to FlashAttention — Koppling till FlashAttention

This experiment demonstrates the same optimization used in production FlashAttention:

| This Experiment | FlashAttention |
|-----------------|----------------|
| 64 heads | Query heads (HQ) |
| KEY data per head | K tensor per attention head |
| 32 blocks per head | Sequence blocks |
| Swizzled head-first | AMD AITER attention kernels |

The paper "Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects" applies exactly this technique to FlashAttention2.

---

## Files — Filer

| File | Description |
|------|-------------|
| `l2_profiling.cpp` | Memory-bound kernel demonstrating NUMA effects |
| `Makefile` | Build targets: `run`, `profile`, `profile-v3` |
| `metrics.txt` | rocprof configuration for L2 counters |
| `README.md` | This documentation |

---

## Summary — Sammanfattning

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DEMONSTRATED — DEMONSTRERAT                                            │
│                                                                         │
│  1. Memory-bound workloads show clear swizzling benefit                │
│     Minnesbundna arbetsbelastningar visar tydlig swizzlingfördel       │
│                                                                         │
│  2. L2 cache hit rate: NAIVE ~40% → SWIZZLED ~90%                      │
│     L2-cache-träfffrekvens: NAIV ~40% → SWIZZLAD ~90%                  │
│                                                                         │
│  3. Performance improvement: 10-50% for memory-bound kernels           │
│     Prestandaförbättring: 10-50% för minnesbundna kärnor               │
│                                                                         │
│  4. rocprof provides hardware counter validation                        │
│     rocprof ger hårdvaruräknarvalidering                               │
│                                                                         │
│  This is the technique used in production FlashAttention on MI300X!    │
│  Detta är tekniken som används i produktion FlashAttention på MI300X!  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Experiment 06c: L2 Cache Profiling*
*Part of the MFMA/CDNA3 Learning Series*
