# Experiment 06: XCD-Aware Kernel Design — MI300X Chiplet Optimization

## Översikt — Overview

This experiment series teaches **NUMA-aware kernel design** for AMD MI300X's multi-chiplet architecture. You'll learn how workgroup scheduling affects L2 cache utilization and how **swizzling** can improve performance by up to 50%.

Denna experimentserie lär dig **NUMA-medveten kärndesign** för AMD MI300X:s multi-chiplet-arkitektur. Du lär dig hur arbetsgruppschemaläggning påverkar L2-cache-utnyttjande och hur **swizzling** kan förbättra prestanda med upp till 50%.

---

## MI300X Architecture — MI300X-arkitektur

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AMD MI300X (8 XCDs)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│   │  XCD 0   │  │  XCD 1   │  │  XCD 2   │  │  XCD 3   │                   │
│   │  38 CUs  │  │  38 CUs  │  │  38 CUs  │  │  38 CUs  │                   │
│   │  4MB L2  │  │  4MB L2  │  │  4MB L2  │  │  4MB L2  │   ← Private L2!   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│   │  XCD 4   │  │  XCD 5   │  │  XCD 6   │  │  XCD 7   │                   │
│   │  38 CUs  │  │  38 CUs  │  │  38 CUs  │  │  38 CUs  │                   │
│   │  4MB L2  │  │  4MB L2  │  │  4MB L2  │  │  4MB L2  │                   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘                   │
│                                                                             │
│   Total: 304 CUs | 32 MB L2 Cache | 192 GB HBM3 | 5.3 TB/s bandwidth       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The NUMA Problem — NUMA-problemet

**Key insight** (Nyckelinsikt): Each XCD has its **own private L2 cache**. Data cached on XCD 0 is **NOT visible** to XCD 1!

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NAIVE SCHEDULING — NAIV SCHEMALÄGGNING                                     │
│                                                                             │
│  Workgroups assigned round-robin: WG0→XCD0, WG1→XCD1, WG2→XCD2, ...        │
│  Arbetsgrupper tilldelas round-robin: WG0→XCD0, WG1→XCD1, WG2→XCD2, ...    │
│                                                                             │
│  If WG0 and WG1 share data, they CAN'T share L2 cache!                     │
│  Om WG0 och WG1 delar data kan de INTE dela L2-cache!                      │
│                                                                             │
│  → L2 cache hit rate: 30-50%                                               │
│  → Each workgroup must load from HBM independently                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  SWIZZLED SCHEDULING — SWIZZLAD SCHEMALÄGGNING                              │
│                                                                             │
│  Workgroups that share data assigned to SAME XCD                           │
│  Arbetsgrupper som delar data tilldelas SAMMA XCD                          │
│                                                                             │
│  WG0, WG1, WG2, ... (sharing data) → all on XCD 0                          │
│                                                                             │
│  → L2 cache hit rate: 80-95%                                               │
│  → First workgroup loads data, others get it from L2!                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Experiment Structure — Experimentstruktur

| Experiment | Topic | Duration | Key Learning |
|------------|-------|----------|--------------|
| **06a** | XCD Discovery | 1-2 hours | Understand chiplet topology, round-robin scheduling |
| **06b** | Swizzled GEMM | 2-3 hours | Implement naive vs swizzled workgroup mapping |
| **06c** | L2 Profiling | 1-2 hours | Measure actual L2 cache hit rates with rocprof |

### Prerequisites — Förkunskaper

- Completed Experiments 01-05 (HIP basics, wavefronts, LDS, MFMA)
- Access to MI300X GPU (gfx942)
- ROCm 6.0+ with rocprof/rocprofv3

---

## Quick Start — Snabbstart

```bash
# Clone or create the experiment directory
# Klona eller skapa experimentkatalogen
mkdir -p ~/mfma/06-xcd-awareness
cd ~/mfma/06-xcd-awareness

# For each sub-experiment, create directory and upload files:
# För varje delexperiment, skapa katalog och ladda upp filer:

# 06a: XCD Discovery
mkdir -p 06a-xcd-discovery
cd 06a-xcd-discovery
# Upload: xcd_discovery.cpp, Makefile, metrics.txt, README.md
make && make run

# 06b: Swizzled GEMM
cd ..
mkdir -p 06b-swizzled-gemm
cd 06b-swizzled-gemm
# Upload: swizzled_gemm.cpp, Makefile, metrics.txt
make && make run-large

# 06c: L2 Profiling
cd ..
mkdir -p 06c-profiling
cd 06c-profiling
# Upload: l2_profiling.cpp, Makefile, metrics.txt
make && make profile
```

---

## Key Concepts — Nyckelkoncept

### 1. Round-Robin Scheduling — Round-Robin-schemaläggning

In SPX mode (single partition), the hardware assigns workgroups to XCDs in order:

```
WG 0 → XCD 0    WG 8  → XCD 0    WG 16 → XCD 0
WG 1 → XCD 1    WG 9  → XCD 1    WG 17 → XCD 1
WG 2 → XCD 2    WG 10 → XCD 2    WG 18 → XCD 2
...             ...              ...
WG 7 → XCD 7    WG 15 → XCD 7    WG 23 → XCD 7
```

**Pattern**: `XCD = workgroup_id % 8`

### 2. Workgroup Swizzling — Arbetsgruppswizzling

Swizzling remaps workgroup IDs so that related workgroups land on the same XCD:

```cpp
// NAIVE: Linear mapping
// NAIV: Linjär mappning
tile_m = wg_id / tiles_n;
tile_n = wg_id % tiles_n;

// SWIZZLED: Group by super-tiles (one per XCD)
// SWIZZLAD: Gruppera efter superplattor (en per XCD)
int super_tile_id = wg_id / tiles_per_super;  // Which XCD
int local_id = wg_id % tiles_per_super;        // Position within XCD
```

### 3. L2 Cache Locality — L2-cache-lokalitet

When workgroups that share data are on the same XCD:
- First workgroup loads data from HBM into L2
- Subsequent workgroups get data from L2 (much faster!)
- L2 hit rate increases from ~40% to ~92%

---

## Expected Results — Förväntade resultat

### Experiment 06a: XCD Discovery

```
Workgroup to XCD mapping:
  WG 0 → XCD 0
  WG 1 → XCD 1
  ...
  WG 7 → XCD 7
  WG 8 → XCD 0  (wraps around)
```

### Experiment 06b: Swizzled GEMM

```
┌───────────────────┬──────────────┬──────────────┐
│  Mapping          │ Time (ms)    │ GFLOPS       │
├───────────────────┼──────────────┼──────────────┤
│  NAIVE            │ ~6.5         │ ~21000       │
│  SWIZZLED         │ ~6.5         │ ~21000       │
└───────────────────┴──────────────┴──────────────┘

Note: GEMM is compute-bound, so timing difference is small.
The L2 benefit is hidden by compute time.
```

### Experiment 06c: L2 Profiling (rocprof)

```
Kernel           │ L2CacheHit
─────────────────┼────────────
NAIVE            │ 30-50%
SWIZZLED         │ 80-95%     ← Big difference!
```

---

## Connection to Production Code — Koppling till produktionskod

This experiment teaches concepts used in:

| This Experiment | Production Equivalent |
|-----------------|----------------------|
| Workgroup swizzling | Composable Kernel `BlockToCTileMap` |
| XCD-aware scheduling | hipBLASLt tile distribution |
| L2 cache optimization | FlashAttention NUMA optimizations |
| Super-tile grouping | AMD AITER attention kernels |

### From the Research Paper — Från forskningsartikeln

This experiment is based on: **"Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects"** (arXiv:2511.02132)

Key findings from the paper:
- L2 cache hit rates: 43% → 92% with NUMA-aware design
- Performance improvement: up to 50% for attention workloads
- The technique applies to any multi-chiplet GPU architecture

---

## Troubleshooting — Felsökning

### rocprof not working

```bash
# Check if rocprof is available
which rocprof
rocprof --version

# If permission denied, may need sudo or add to group
sudo usermod -a -G video $USER

# Try rocprofv3 instead
rocprofv3 --hip-trace ./l2_profiling
```

### No L2CacheHit in results

```bash
# List available counters for gfx942
rocprof --list-basic | grep -i l2
rocprof --list-basic | grep -i cache

# Some counters may have different names
# Try: TCP_TCC_HIT_sum, TCP_TCC_MISS_sum
```

### Similar performance for naive vs swizzled

This is expected for compute-bound workloads like GEMM. The benefit shows in:
1. L2 cache hit rate (measured with rocprof)
2. Memory-bound workloads (like attention in 06c)
3. Larger problem sizes with more memory pressure

---

## Next Steps — Nästa steg

After completing Experiment 06, you can:

1. **Study Composable Kernel** — See how CK implements tile mapping
2. **Contribute to vLLM ROCm** — Apply NUMA optimizations to inference
3. **Experiment with FlashAttention** — The original use case for these techniques
4. **Try FP8/INT4 quantization** — Next frontier for MI300X optimization

---

## References — Referenser

- Paper: [Optimizing Attention on GPUs (2511.02132)](https://arxiv.org/abs/2511.02132)
- AMD MI300X Architecture Guide
- ROCm Profiler Documentation
- Composable Kernel Source: `library/include/ck/tile/`

---

## File Structure — Filstruktur

```
06-xcd-awareness/
├── README.md                    # This file / Denna fil
├── 06a-xcd-discovery/
│   ├── README.md
│   ├── xcd_discovery.cpp
│   ├── Makefile
│   └── metrics.txt
├── 06b-swizzled-gemm/
│   ├── README.md
│   ├── swizzled_gemm.cpp
│   ├── Makefile
│   └── metrics.txt
└── 06c-profiling/
    ├── README.md
    ├── l2_profiling.cpp
    ├── Makefile
    └── metrics.txt
```

---

*Experiment 06: XCD-Aware Kernel Design*
*For AMD MI300X (gfx942) / CDNA3 Architecture*
*January 2025*
