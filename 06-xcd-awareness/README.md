# Experiment 06a: XCD Discovery — Förstå MI300X Chiplet-arkitektur

## Overview — Översikt

This experiment teaches you the foundational concepts of **NUMA-aware kernel design** on AMD MI300X. Du lär dig (you'll learn) how workgroups are distributed across the 8 XCDs and why this matters för prestanda (for performance).

## Key Concepts — Nyckelkoncept

### What is an XCD? — Vad är en XCD?

**XCD** = **Accelerator Complex Die** — a chiplet containing:
- 38 Compute Units (CUs)
- 4 MB private L2 cache
- Memory controllers

MI300X har (has) **8 XCDs** = 304 CUs total, 32 MB L2 cache total.

### The NUMA Problem — NUMA-problemet

```
┌─────────────────────────────────────────────────────────────┐
│  XCD 0 (L2 = 4MB)    │    XCD 1 (L2 = 4MB)                 │
│  ┌─────────────────┐ │    ┌─────────────────┐              │
│  │ Workgroup 0     │ │    │ Workgroup 1     │              │
│  │ Loads Key[0]    │ │    │ Loads Key[0]    │ ← SAME data! │
│  │ → Cached in L2  │ │    │ → NOT in L2!    │ ← Must fetch │
│  └─────────────────┘ │    │ → Fetch from HBM│    from HBM! │
│                      │    └─────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Problem**: Om (if) workgroups that share data end up on different XCDs, they **cannot share L2 cache**. Each must fetch from HBM independently!

**Resultat (Result)**: L2 cache hit rate drops from 90%+ to 1-10%.

### The Solution: Swizzling — Lösningen: Swizzling

**Swizzling** = Remapping workgroup IDs so that related workgroups land on the same XCD.

```
NAIVE:     WG0→XCD0, WG1→XCD1, WG2→XCD2, ... (spreads data!)
SWIZZLED:  WG0→XCD0, WG1→XCD0, WG2→XCD0, WG8→XCD1, ... (keeps data local!)
```

## Building & Running — Bygga & Köra

### Step 1: Build — Bygg

```bash
# On your MI300X instance
# På din MI300X-instans
cd ~/mfma/06-xcd-awareness/

# Or copy from this experiment
# Eller kopiera från detta experiment
mkdir -p ~/mfma/06-xcd-awareness/
cp xcd_discovery.cpp Makefile metrics.txt ~/mfma/06-xcd-awareness/
cd ~/mfma/06-xcd-awareness/

# Build
make
```

### Step 2: Run Basic Discovery — Kör grundläggande upptäckt

```bash
make run
```

**Expected output / Förväntad utdata:**
- Device information showing gfx942 (MI300X)
- Architecture diagram showing 8 XCDs
- Round-robin workgroup assignment table
- Comparison of naive vs swizzled mapping

### Step 3: Profile with rocprof — Profilera med rocprof

```bash
make profile
```

This runs the kernel multiple times to collect L2 cache metrics.
Detta kör kärnan flera gånger för att samla L2-cache-metriker.

**Look for / Leta efter:**
- `L2CacheHit` — Higher is better (högre är bättre)
- Compare naive kernel runs vs swizzled kernel runs

## Understanding the Output — Förstå utdatan

### Naive vs Swizzled Comparison

```
NAIVE MAPPING                    SWIZZLED MAPPING
──────────────                   ─────────────────
WG │ Head │ Block │ XCD          WG │ Head │ Block │ XCD
 0 │   0  │   0   │  0            0 │   0  │   0   │  0
 1 │   0  │   1   │  1  ← Bad!    1 │   0  │   1   │  0  ← Same XCD!
 2 │   0  │   2   │  2  ← Bad!    2 │   0  │   2   │  0  ← Same XCD!
 3 │   0  │   3   │  3  ← Bad!    3 │   0  │   3   │  0  ← Same XCD!
```

**Naive**: Blocks of Head 0 spread across XCDs 0,1,2,3 → No L2 sharing!
**Swizzled**: All blocks of Head 0 stay on XCD 0 → Perfect L2 sharing!

## Exercises — Övningar

### Exercise 1: Modify Number of Heads — Modifiera antal huvuden

Change `num_heads` from 8 to 16 or 32 and observe:
- How does the mapping change?
- When does swizzling still help?

### Exercise 2: Check rocprof Counters — Kontrollera rocprof-räknare

Run `rocprof --list-basic` to see available counters for gfx942.
Kör `rocprof --list-basic` för att se tillgängliga räknare för gfx942.

```bash
rocprof --list-basic | grep -i l2
rocprof --list-basic | grep -i cache
```

### Exercise 3: Trace with rocprofv3 — Spåra med rocprofv3

```bash
make profile-v3
```

This creates detailed trace files you can view in Perfetto.
Detta skapar detaljerade spårfiler du kan visa i Perfetto.

## Connection to Composable Kernel — Koppling till Composable Kernel

This experiment teaches concepts used in CK's GEMM and attention kernels:

| This Experiment | CK Equivalent |
|-----------------|---------------|
| Workgroup swizzling | `BlockToCTileMap` |
| XCD-aware scheduling | `GridwiseGemm` tile distribution |
| L2 cache optimisation | `BlockwiseTensorSliceTransfer` patterns |

## Next Steps — Nästa steg

After completing this experiment, proceed to:
- **06b**: Implement naive vs swizzled GEMM kernels
- **06c**: Profile and measure actual L2 cache hit rates

## References — Referenser

- Paper: "Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects" (2511.02132v1.pdf)
- AMD ROCm Documentation: MI300X workload optimization
- Composable Kernel source: `library/include/ck/tile/`

---

*Experiment 06a — Created January 2026*
*For AMD MI300X (gfx942) / CDNA3 Architecture*