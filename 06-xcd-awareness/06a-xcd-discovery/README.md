# Experiment 06a: XCD Discovery — Förstå MI300X Chiplet-arkitektur

## Syfte — Purpose

Learn the foundational concepts of MI300X's multi-chiplet architecture:
- How 8 XCDs (Accelerator Complex Dies) are organized
- How workgroups are distributed across XCDs (round-robin)
- Why this creates NUMA effects for L2 cache access

Lär dig de grundläggande koncepten för MI300X:s multi-chiplet-arkitektur:
- Hur 8 XCD:er (Accelerator Complex Dies) är organiserade
- Hur arbetsgrupper fördelas över XCD:er (round-robin)
- Varför detta skapar NUMA-effekter för L2-cache-åtkomst

---

## Key Concepts — Nyckelkoncept

### What is an XCD? — Vad är en XCD?

**XCD** = **Accelerator Complex Die** — a chiplet containing:

| Component | Specification |
|-----------|---------------|
| Compute Units | 38 per XCD |
| L2 Cache | 4 MB **private** per XCD |
| Stream Processors | 64 per CU |

MI300X has **8 XCDs** = 304 CUs total, 32 MB L2 cache total.

### Round-Robin Scheduling — Round-Robin-schemaläggning

In SPX mode, workgroups are assigned to XCDs in order:

```
Workgroup ID │ XCD Assignment
─────────────┼────────────────
     0       │      0
     1       │      1
     2       │      2
     3       │      3
     4       │      4
     5       │      5
     6       │      6
     7       │      7
     8       │      0  ← Wraps around / Går runt
     9       │      1
    ...      │     ...
```

**Formula**: `XCD = workgroup_id % 8`

### The NUMA Problem — NUMA-problemet

```
┌────────────────────────────────────────────────────────────────────────┐
│  XCD 0                          │  XCD 1                              │
│  ┌─────────────────────────┐    │  ┌─────────────────────────┐       │
│  │ Workgroup 0             │    │  │ Workgroup 1             │       │
│  │ Loads Data[0]           │    │  │ Needs Data[0] too!      │       │
│  │ → Stored in XCD 0's L2  │    │  │ → NOT in XCD 1's L2!    │       │
│  └─────────────────────────┘    │  │ → Must load from HBM    │       │
│                                 │  └─────────────────────────┘       │
│  L2 Cache: Has Data[0] ✓        │  L2 Cache: Empty ✗                 │
└────────────────────────────────────────────────────────────────────────┘

Problem: WG0 and WG1 share data but are on different XCDs!
         They cannot share L2 cache.
         
Problem: WG0 och WG1 delar data men är på olika XCD:er!
         De kan inte dela L2-cache.
```

---

## Building & Running — Bygga & Köra

### Step 1: Build — Bygg

```bash
cd ~/mfma/06-xcd-awareness/06a-xcd-discovery
make
```

Expected output:
```
hipcc --offload-arch=gfx942 -O3 -o xcd_discovery xcd_discovery.cpp
✓ Built xcd_discovery for gfx942
```

### Step 2: Run — Kör

```bash
make run
```

### Step 3: Profile (optional) — Profilera (valfritt)

```bash
make profile
```

---

## Understanding the Output — Förstå utdatan

### 1. Architecture Diagram

The program displays the MI300X architecture with 8 XCDs.

### 2. Round-Robin Table

Shows how workgroups map to XCDs:

```
WG ID │ XCD │ Threads
──────┼─────┼────────
    0 │   0 │      64
    1 │   1 │      64
    2 │   2 │      64
  ...
```

### 3. Naive vs Swizzled Comparison

Side-by-side comparison of mapping strategies:

```
┌────────────────────────────────────┬────────────────────────────────────┐
│         NAIVE MAPPING              │         SWIZZLED MAPPING           │
├────────────────────────────────────┼────────────────────────────────────┤
│  WG │ Head │ Block │ XCD          │  WG │ Head │ Block │ XCD           │
│   0 │   0  │   0   │  0           │   0 │   0  │   0   │  0            │
│   1 │   0  │   1   │  1  ← Bad!   │   1 │   0  │   1   │  0  ← Good!   │
│   2 │   0  │   2   │  2  ← Bad!   │   2 │   0  │   2   │  0  ← Good!   │
└────────────────────────────────────┴────────────────────────────────────┘
```

**Naive**: Head 0's blocks spread across XCDs 0,1,2,3,4,5,6,7 — No L2 sharing!
**Swizzled**: Head 0's blocks all on XCD 0 — Perfect L2 sharing!

---

## Exercises — Övningar

### Exercise 1: Modify Number of Heads

Edit the code to change `num_heads` from 8 to 16 or 32:

```cpp
const int num_heads = 16;  // Change from 8
```

Observe how the mapping changes. Questions to answer:
- How many heads per XCD with swizzled mapping?
- Does swizzling still help with 32 heads?

### Exercise 2: Explore rocprof Counters

```bash
# List available counters for gfx942
rocprof --list-basic | grep -i l2
rocprof --list-basic | grep -i cache
rocprof --list-basic | grep -i tcc
```

### Exercise 3: Visualize with rocprofv3

```bash
make profile-v3
# Creates trace files in results_v3/
# View at https://ui.perfetto.dev/
```

---

## Code Walkthrough — Kodgenomgång

### Key Function: Round-Robin XCD Assignment

```cpp
// In k_discover_xcd_assignment kernel:
xcd_assignments[wg_id] = wg_id % NUM_XCDS;

// This mirrors what the hardware does automatically
// Detta speglar vad hårdvaran gör automatiskt
```

### Key Function: Swizzled Mapping

```cpp
// Group heads so each XCD handles (num_heads / 8) heads
int heads_per_xcd = (num_heads + NUM_XCDS - 1) / NUM_XCDS;
int blocks_per_xcd = heads_per_xcd * blocks_per_head;

// Which XCD should this workgroup run on?
int target_xcd = wg_id / blocks_per_xcd;
```

---

## Connection to Next Experiments — Koppling till nästa experiment

After understanding XCD distribution, you'll:

| Next Experiment | What You'll Learn |
|-----------------|-------------------|
| **06b** | Implement actual GEMM kernels with both mappings |
| **06c** | Measure L2 cache hit rates with rocprof |

---

## Files in This Experiment — Filer i detta experiment

| File | Purpose |
|------|---------|
| `xcd_discovery.cpp` | Main source with educational comments |
| `Makefile` | Build targets: `run`, `profile`, `clean` |
| `metrics.txt` | rocprof configuration for L2 counters |
| `README.md` | This documentation |

---

## Summary — Sammanfattning

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KEY TAKEAWAY — HUVUDPOÄNG                                              │
│                                                                         │
│  MI300X has 8 XCDs, each with private 4MB L2 cache.                    │
│  MI300X har 8 XCD:er, var och en med privat 4MB L2-cache.              │
│                                                                         │
│  Default round-robin scheduling spreads related workgroups across       │
│  different XCDs, preventing L2 cache sharing.                          │
│                                                                         │
│  Standard round-robin-schemaläggning sprider relaterade arbetsgrupper   │
│  över olika XCD:er, vilket förhindrar L2-cache-delning.                │
│                                                                         │
│  Solution: Swizzle workgroup IDs to keep related work on same XCD.     │
│  Lösning: Swizzla arbetsgrupp-ID:n för att hålla relaterat arbete på   │
│           samma XCD.                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Experiment 06a: XCD Discovery*
*Part of the MFMA/CDNA3 Learning Series*
