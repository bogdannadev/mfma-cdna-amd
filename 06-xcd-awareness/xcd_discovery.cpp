/**
 * =============================================================================
 * Experiment 06a: XCD Discovery — Förstå MI300X Chiplet-arkitektur
 * =============================================================================
 *
 * SYFTE (PURPOSE):
 *   Understand how workgroups are distributed across the 8 XCDs in MI300X.
 *   Learn the foundation for NUMA-aware kernel design.
 *   Förstå hur arbetsgrupper fördelas över de 8 XCD:erna i MI300X.
 *
 * MI300X ARCHITECTURE — ARKITEKTUR:
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                        MI300X (8 XCDs)                              │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
 *   │  │  XCD 0  │ │  XCD 1  │ │  XCD 2  │ │  XCD 3  │    4 XCDs        │
 *   │  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │    on IOD 0-1    │
 *   │  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │                   │
 *   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
 *   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
 *   │  │  XCD 4  │ │  XCD 5  │ │  XCD 6  │ │  XCD 7  │    4 XCDs        │
 *   │  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │    on IOD 2-3    │
 *   │  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │                   │
 *   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
 *   │                                                                     │
 *   │  Total: 304 CUs, 32 MB L2 Cache, 192 GB HBM3                       │
 *   │  Totalt: 304 beräkningsenheter, 32 MB L2-cache, 192 GB HBM3        │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * KEY CONCEPT — NYCKELKONCEPT:
 *
 *   In SPX mode (Single Partition), workgroups are distributed ROUND-ROBIN
 *   across XCDs. This means:
 *   I SPX-läge fördelas arbetsgrupper ROUND-ROBIN över XCD:er. Detta innebär:
 *
 *     Workgroup 0 → XCD 0
 *     Workgroup 1 → XCD 1
 *     Workgroup 2 → XCD 2
 *     ...
 *     Workgroup 7 → XCD 7
 *     Workgroup 8 → XCD 0  (wraps around / går runt)
 *     Workgroup 9 → XCD 1
 *     ...
 *
 *   PROBLEM: If workgroups 0 and 8 share data, they're on the SAME XCD
 *            and can share L2 cache. But workgroups 0 and 1 are on
 *            DIFFERENT XCDs and cannot share L2 cache!
 *
 *   PROBLEM: Om arbetsgrupp 0 och 8 delar data, är de på SAMMA XCD
 *            och kan dela L2-cache. Men arbetsgrupp 0 och 1 är på
 *            OLIKA XCD:er och kan inte dela L2-cache!
 *
 * WHY THIS MATTERS — VARFÖR DETTA ÄR VIKTIGT:
 *
 *   L2 cache hit rates can vary from 1% to 97% depending on how you
 *   organise your workgroups! (See paper: 2511.02132v1.pdf)
 *   L2-cache-träfffrekvensen kan variera från 1% till 97% beroende på
 *   hur du organiserar dina arbetsgrupper!
 *
 * BUILD — KOMPILERA:
 *   hipcc --offload-arch=gfx942 -O3 -o xcd_discovery xcd_discovery.cpp
 *
 * RUN — KÖR:
 *   ./xcd_discovery
 *
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <cstdio>

/**
 * HIP_CHECK macro — Felhanteringsmakro
 *
 * Always check HIP API calls for errors!
 * Kontrollera alltid HIP API-anrop för fel!
 */
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/**
 * NUM_XCDS — Number of XCDs in MI300X
 * Antal XCD:er i MI300X
 *
 * This is a constant for the MI300X architecture.
 * Detta är en konstant för MI300X-arkitekturen.
 */
constexpr int NUM_XCDS = 8;

/**
 * CUS_PER_XCD — Compute Units per XCD
 * Beräkningsenheter per XCD
 *
 * MI300X has 38 active CUs per XCD (40 total, 2 disabled for yield).
 * MI300X har 38 aktiva CU:er per XCD (40 totalt, 2 inaktiverade för utbyte).
 */
constexpr int CUS_PER_XCD = 38;

/**
 * =============================================================================
 * KERNEL: k_discover_xcd_assignment
 * =============================================================================
 *
 * This kernel records which "logical XCD" each workgroup runs on.
 * Denna kernel registrerar vilken "logisk XCD" varje arbetsgrupp körs på.
 *
 * HOW IT WORKS — HUR DET FUNGERAR:
 *
 *   We cannot directly query "which XCD am I on?" in HIP. However, we CAN
 *   infer it from the workgroup ID using the round-robin scheduling pattern.
 *   Vi kan inte direkt fråga "vilken XCD är jag på?" i HIP. Men vi KAN
 *   härleda det från arbetsgrupp-ID:t med round-robin-schemaläggningsmönstret.
 *
 *   Logical XCD = workgroup_id % NUM_XCDS
 *   Logisk XCD = arbetsgrupp_id % NUM_XCDS
 *
 * PARAMETERS — PARAMETRAR:
 *   @param workgroup_ids   Output: workgroup ID for each workgroup
 *                          Utdata: arbetsgrupp-ID för varje arbetsgrupp
 *   @param xcd_assignments Output: inferred XCD assignment (0-7)
 *                          Utdata: härledd XCD-tilldelning (0-7)
 *   @param thread_counts   Output: number of threads that ran in each workgroup
 *                          Utdata: antal trådar som körde i varje arbetsgrupp
 *   @param num_workgroups  Total number of workgroups to launch
 *                          Totalt antal arbetsgrupper att starta
 */
__global__ void k_discover_xcd_assignment(
    int* workgroup_ids,
    int* xcd_assignments,
    int* thread_counts,
    int num_workgroups
) {
    /**
     * Get workgroup (block) ID — Hämta arbetsgrupp-ID
     *
     * In HIP/CUDA terminology:
     *   blockIdx.x = workgroup ID (arbetsgrupp-ID)
     *   threadIdx.x = thread ID within workgroup (tråd-ID inom arbetsgruppen)
     *
     * I HIP/CUDA-terminologi:
     *   blockIdx.x = arbetsgrupp-ID
     *   threadIdx.x = tråd-ID inom arbetsgruppen
     */
    int wg_id = blockIdx.x;
    int tid = threadIdx.x;

    /**
     * Only thread 0 of each workgroup records the data
     * Endast tråd 0 i varje arbetsgrupp registrerar data
     *
     * This avoids race conditions (kapplöpningsförhållanden) where multiple
     * threads try to write to the same memory location.
     */
    if (tid == 0 && wg_id < num_workgroups) {
        /**
         * Record the workgroup ID — Registrera arbetsgrupp-ID
         */
        workgroup_ids[wg_id] = wg_id;

        /**
         * Infer XCD assignment using round-robin pattern
         * Härled XCD-tilldelning med round-robin-mönster
         *
         * In SPX mode, the hardware scheduler assigns workgroups to XCDs
         * in a simple round-robin fashion: 0,1,2,3,4,5,6,7,0,1,2,3,...
         * I SPX-läge tilldelar hårdvaruschemaläggaren arbetsgrupper till
         * XCD:er i ett enkelt round-robin-mönster: 0,1,2,3,4,5,6,7,0,1,2,3,...
         */
        xcd_assignments[wg_id] = wg_id % NUM_XCDS;

        /**
         * Record block size — Registrera blockstorlek
         */
        thread_counts[wg_id] = blockDim.x;
    }
}

/**
 * =============================================================================
 * KERNEL: k_demonstrate_naive_vs_swizzled
 * =============================================================================
 *
 * This kernel demonstrates the difference between NAIVE and SWIZZLED
 * workgroup assignment patterns.
 * Denna kernel demonstrerar skillnaden mellan NAIVA och SWIZZLADE
 * arbetsgruppstilldelningsmönster.
 *
 * SCENARIO — SCENARIO:
 *   Imagine a 2D grid of tiles (like in GEMM or attention).
 *   Föreställ dig ett 2D-rutnät av plattor (som i GEMM eller attention).
 *
 *   Grid: num_heads × num_blocks_per_head
 *   Rutnät: num_heads × num_blocks_per_head
 *
 * NAIVE MAPPING — NAIV MAPPNING:
 *   Linear workgroup ID → (head, block) in row-major order
 *   Linjärt arbetsgrupp-ID → (huvud, block) i radordning
 *
 *     WG 0 → Head 0, Block 0 → XCD 0
 *     WG 1 → Head 0, Block 1 → XCD 1
 *     WG 2 → Head 0, Block 2 → XCD 2
 *     ...
 *     WG 8 → Head 1, Block 0 → XCD 0
 *
 *   PROBLEM: Blocks of the SAME head end up on DIFFERENT XCDs!
 *            They share K,V data but can't share L2 cache!
 *   PROBLEM: Block av SAMMA huvud hamnar på OLIKA XCD:er!
 *            De delar K,V-data men kan inte dela L2-cache!
 *
 * SWIZZLED MAPPING — SWIZZLAD MAPPNING:
 *   Remap workgroup IDs so that blocks of the same head stay on the same XCD.
 *   Ommappa arbetsgrupp-ID:n så att block av samma huvud stannar på samma XCD.
 *
 *     WG 0 → Head 0, Block 0 → XCD 0
 *     WG 1 → Head 0, Block 1 → XCD 0  (SAME XCD! / SAMMA XCD!)
 *     WG 2 → Head 0, Block 2 → XCD 0
 *     ...
 *     WG 8 → Head 1, Block 0 → XCD 1  (Different head → different XCD)
 *
 *   RESULT: L2 cache hit rate 80-97% instead of 1%!
 *   RESULTAT: L2-cache-träfffrekvens 80-97% istället för 1%!
 */
__global__ void k_demonstrate_naive_vs_swizzled(
    int* naive_head,
    int* naive_block,
    int* naive_xcd,
    int* swizzled_head,
    int* swizzled_block,
    int* swizzled_xcd,
    int num_heads,
    int blocks_per_head
) {
    int wg_id = blockIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) {
        /**
         * =================================================================
         * NAIVE MAPPING — NAIV MAPPNING
         * =================================================================
         *
         * Simple linear mapping: iterate through blocks, then heads.
         * Enkel linjär mappning: iterera genom block, sedan huvuden.
         *
         * Formula / Formel:
         *   head = wg_id / blocks_per_head
         *   block = wg_id % blocks_per_head
         */
        int n_head = wg_id / blocks_per_head;
        int n_block = wg_id % blocks_per_head;
        int n_xcd = wg_id % NUM_XCDS;  // Round-robin XCD assignment

        naive_head[wg_id] = n_head;
        naive_block[wg_id] = n_block;
        naive_xcd[wg_id] = n_xcd;

        /**
         * =================================================================
         * SWIZZLED HEAD-FIRST MAPPING — SWIZZLAD HUVUD-FÖRST MAPPNING
         * =================================================================
         *
         * From paper (2511.02132v1.pdf), Figure 11:
         * Från artikeln (2511.02132v1.pdf), Figur 11:
         *
         * The goal is to assign ALL blocks of the same head to the SAME XCD.
         * Målet är att tilldela ALLA block av samma huvud till SAMMA XCD.
         *
         * Key insight — Nyckelinsikt:
         *   heads_per_xcd = num_heads / NUM_XCDS
         *   We group heads so each XCD handles (num_heads / 8) heads.
         *   Vi grupperar huvuden så att varje XCD hanterar (num_heads / 8) huvuden.
         *
         * Swizzling formula / Swizzlingsformel:
         *   target_xcd = head / heads_per_xcd
         *   Within that XCD, we process all blocks of each head sequentially.
         *   Inom den XCD:n bearbetar vi alla block av varje huvud sekventiellt.
         */
        int heads_per_xcd = (num_heads + NUM_XCDS - 1) / NUM_XCDS;  // Ceiling division
        int blocks_per_xcd = heads_per_xcd * blocks_per_head;

        /**
         * Calculate which XCD this workgroup should run on
         * Beräkna vilken XCD denna arbetsgrupp ska köras på
         */
        int target_xcd = wg_id / blocks_per_xcd;
        if (target_xcd >= NUM_XCDS) target_xcd = NUM_XCDS - 1;  // Clamp

        /**
         * Calculate head and block within this XCD's allocation
         * Beräkna huvud och block inom denna XCD:s tilldelning
         */
        int local_id = wg_id % blocks_per_xcd;
        int s_head = target_xcd * heads_per_xcd + (local_id / blocks_per_head);
        int s_block = local_id % blocks_per_head;

        // Clamp head to valid range / Begränsa huvud till giltigt intervall
        if (s_head >= num_heads) s_head = num_heads - 1;

        swizzled_head[wg_id] = s_head;
        swizzled_block[wg_id] = s_block;
        swizzled_xcd[wg_id] = target_xcd;
    }
}

/**
 * =============================================================================
 * KERNEL: k_cache_locality_demo
 * =============================================================================
 *
 * This kernel demonstrates cache locality effects by having workgroups
 * access shared data.
 * Denna kernel demonstrerar cache-lokalitetseffekter genom att låta
 * arbetsgrupper komma åt delad data.
 *
 * SETUP — UPPSÄTTNING:
 *   - We have NUM_HEADS "attention heads"
 *   - Each head has a KEY buffer that all its blocks share
 *   - Vi har NUM_HEADS "attention-huvuden"
 *   - Varje huvud har en KEY-buffert som alla dess block delar
 *
 * MEASUREMENT — MÄTNING:
 *   - Run with rocprof to measure L2 cache hit rate
 *   - Compare naive vs swizzled mapping
 *   - Kör med rocprof för att mäta L2-cache-träfffrekvens
 *   - Jämför naiv vs swizzlad mappning
 */
__global__ void k_cache_locality_demo(
    float* shared_keys,      // [num_heads][key_size] - shared data per head
    float* output,           // [num_workgroups] - output per workgroup
    int num_heads,
    int blocks_per_head,
    int key_size,
    int use_swizzling        // 0 = naive, 1 = swizzled
) {
    int wg_id = blockIdx.x;
    int tid = threadIdx.x;

    int head_idx;

    if (use_swizzling) {
        /**
         * SWIZZLED: Calculate head using swizzled mapping
         * SWIZZLAD: Beräkna huvud med swizzlad mappning
         */
        int heads_per_xcd = (num_heads + NUM_XCDS - 1) / NUM_XCDS;
        int blocks_per_xcd = heads_per_xcd * blocks_per_head;
        int target_xcd = wg_id / blocks_per_xcd;
        if (target_xcd >= NUM_XCDS) target_xcd = NUM_XCDS - 1;
        int local_id = wg_id % blocks_per_xcd;
        head_idx = target_xcd * heads_per_xcd + (local_id / blocks_per_head);
        if (head_idx >= num_heads) head_idx = num_heads - 1;
    } else {
        /**
         * NAIVE: Simple linear mapping
         * NAIV: Enkel linjär mappning
         */
        head_idx = wg_id / blocks_per_head;
        if (head_idx >= num_heads) head_idx = num_heads - 1;
    }

    /**
     * Access the KEY data for this head
     * Kom åt KEY-datan för detta huvud
     *
     * All workgroups processing the same head will access the same KEY data.
     * If they're on the same XCD, this data will be in L2 cache!
     * Alla arbetsgrupper som bearbetar samma huvud kommer att komma åt samma KEY-data.
     * Om de är på samma XCD kommer denna data att vara i L2-cache!
     */
    float* key_ptr = shared_keys + head_idx * key_size;

    /**
     * Simulate reading KEY data (like in attention)
     * Simulera läsning av KEY-data (som i attention)
     */
    float sum = 0.0f;
    for (int i = tid; i < key_size; i += blockDim.x) {
        sum += key_ptr[i];
    }

    /**
     * Reduce within workgroup (simplified)
     * Reducera inom arbetsgrupp (förenklad)
     */
    __shared__ float partial_sums[256];
    partial_sums[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x && i < 256; i++) {
            total += partial_sums[i];
        }
        output[wg_id] = total;
    }
}

/**
 * =============================================================================
 * Print functions — Utskriftsfunktioner
 * =============================================================================
 */

void print_header() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Experiment 06a: XCD Discovery — Förstå MI300X Chiplet-arkitektur            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void print_architecture_info() {
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  MI300X Architecture Overview — Arkitekturöversikt                           │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│                                                                              │\n");
    printf("│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                            │\n");
    printf("│  │  XCD 0  │ │  XCD 1  │ │  XCD 2  │ │  XCD 3  │   ← 4 XCDs on IOD 0-1     │\n");
    printf("│  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │                            │\n");
    printf("│  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │   ← Private L2 per XCD    │\n");
    printf("│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                            │\n");
    printf("│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                            │\n");
    printf("│  │  XCD 4  │ │  XCD 5  │ │  XCD 6  │ │  XCD 7  │   ← 4 XCDs on IOD 2-3     │\n");
    printf("│  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │                            │\n");
    printf("│  │  4MB L2 │ │  4MB L2 │ │  4MB L2 │ │  4MB L2 │                            │\n");
    printf("│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                            │\n");
    printf("│                                                                              │\n");
    printf("│  Key Point (Nyckelpunkt):                                                    │\n");
    printf("│  Each XCD has its OWN L2 cache. Data cached on XCD 0 is NOT visible         │\n");
    printf("│  to XCD 1! This is the NUMA effect we must optimize for.                    │\n");
    printf("│  Varje XCD har sin EGEN L2-cache. Data cachad på XCD 0 är INTE synlig       │\n");
    printf("│  för XCD 1! Detta är NUMA-effekten vi måste optimera för.                   │\n");
    printf("│                                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

void print_round_robin_explanation() {
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Round-Robin Scheduling — Round-Robin-schemaläggning                         │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│                                                                              │\n");
    printf("│  In SPX mode, workgroups are assigned to XCDs in round-robin order:         │\n");
    printf("│  I SPX-läge tilldelas arbetsgrupper till XCD:er i round-robin-ordning:      │\n");
    printf("│                                                                              │\n");
    printf("│    WG 0 → XCD 0    WG 8  → XCD 0    WG 16 → XCD 0                           │\n");
    printf("│    WG 1 → XCD 1    WG 9  → XCD 1    WG 17 → XCD 1                           │\n");
    printf("│    WG 2 → XCD 2    WG 10 → XCD 2    WG 18 → XCD 2                           │\n");
    printf("│    WG 3 → XCD 3    WG 11 → XCD 3    WG 19 → XCD 3                           │\n");
    printf("│    WG 4 → XCD 4    WG 12 → XCD 4    WG 20 → XCD 4                           │\n");
    printf("│    WG 5 → XCD 5    WG 13 → XCD 5    WG 21 → XCD 5                           │\n");
    printf("│    WG 6 → XCD 6    WG 14 → XCD 6    WG 22 → XCD 6                           │\n");
    printf("│    WG 7 → XCD 7    WG 15 → XCD 7    WG 23 → XCD 7                           │\n");
    printf("│                                                                              │\n");
    printf("│  Pattern: XCD = workgroup_id %% 8                                            │\n");
    printf("│  Mönster: XCD = arbetsgrupp_id %% 8                                          │\n");
    printf("│                                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

int main() {
    print_header();

    /**
     * Query device properties — Fråga enhetsegenskaper
     */
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    printf("Device Information — Enhetsinformation:\n");
    printf("  Name / Namn:           %s\n", props.name);
    printf("  Architecture / Arkitektur: %s\n", props.gcnArchName);
    printf("  Compute Units / Beräkningsenheter: %d\n", props.multiProcessorCount);
    printf("  L2 Cache Size / L2-cachestorlek: %d KB\n", props.l2CacheSize / 1024);
    printf("  Max Threads per Block / Max trådar per block: %d\n", props.maxThreadsPerBlock);
    printf("\n");

    print_architecture_info();
    print_round_robin_explanation();

    /**
     * ==========================================================================
     * TEST 1: Basic XCD Assignment Discovery
     * TEST 1: Grundläggande XCD-tilldelningsupptäckt
     * ==========================================================================
     */
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Test 1: XCD Assignment Discovery — XCD-tilldelningsupptäckt                 │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");

    const int num_workgroups = 32;
    const int threads_per_block = 64;  // One wavefront / En wavefront

    int *d_wg_ids, *d_xcd_assignments, *d_thread_counts;
    int *h_wg_ids, *h_xcd_assignments, *h_thread_counts;

    size_t size = num_workgroups * sizeof(int);

    // Allocate device memory / Allokera enhetsminne
    HIP_CHECK(hipMalloc(&d_wg_ids, size));
    HIP_CHECK(hipMalloc(&d_xcd_assignments, size));
    HIP_CHECK(hipMalloc(&d_thread_counts, size));

    // Allocate host memory / Allokera värdminne
    h_wg_ids = (int*)malloc(size);
    h_xcd_assignments = (int*)malloc(size);
    h_thread_counts = (int*)malloc(size);

    // Launch kernel / Starta kernel
    hipLaunchKernelGGL(k_discover_xcd_assignment,
                       dim3(num_workgroups), dim3(threads_per_block), 0, 0,
                       d_wg_ids, d_xcd_assignments, d_thread_counts, num_workgroups);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy results back / Kopiera resultat tillbaka
    HIP_CHECK(hipMemcpy(h_wg_ids, d_wg_ids, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_xcd_assignments, d_xcd_assignments, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_thread_counts, d_thread_counts, size, hipMemcpyDeviceToHost));

    printf("\n  Workgroup to XCD mapping (first 16):\n");
    printf("  Arbetsgrupp till XCD-mappning (första 16):\n\n");
    printf("  WG ID │ XCD │ Threads\n");
    printf("  ──────┼─────┼────────\n");
    for (int i = 0; i < 16 && i < num_workgroups; i++) {
        printf("  %5d │ %3d │ %7d\n", h_wg_ids[i], h_xcd_assignments[i], h_thread_counts[i]);
    }
    printf("\n");

    // Count workgroups per XCD / Räkna arbetsgrupper per XCD
    int wg_per_xcd[NUM_XCDS] = {0};
    for (int i = 0; i < num_workgroups; i++) {
        wg_per_xcd[h_xcd_assignments[i]]++;
    }

    printf("  Workgroups per XCD / Arbetsgrupper per XCD:\n");
    for (int i = 0; i < NUM_XCDS; i++) {
        printf("    XCD %d: %d workgroups\n", i, wg_per_xcd[i]);
    }
    printf("\n");

    // Cleanup test 1 / Städa upp test 1
    HIP_CHECK(hipFree(d_wg_ids));
    HIP_CHECK(hipFree(d_xcd_assignments));
    HIP_CHECK(hipFree(d_thread_counts));
    free(h_wg_ids);
    free(h_xcd_assignments);
    free(h_thread_counts);

    /**
     * ==========================================================================
     * TEST 2: Naive vs Swizzled Mapping Comparison
     * TEST 2: Jämförelse mellan naiv och swizzlad mappning
     * ==========================================================================
     */
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Test 2: Naive vs Swizzled Mapping — Naiv vs Swizzlad Mappning               │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");

    const int num_heads = 8;
    const int blocks_per_head = 4;
    const int total_wgs = num_heads * blocks_per_head;

    printf("\n  Configuration / Konfiguration:\n");
    printf("    Heads / Huvuden: %d\n", num_heads);
    printf("    Blocks per head / Block per huvud: %d\n", blocks_per_head);
    printf("    Total workgroups / Totalt arbetsgrupper: %d\n", total_wgs);
    printf("\n");

    int *d_naive_head, *d_naive_block, *d_naive_xcd;
    int *d_swizzled_head, *d_swizzled_block, *d_swizzled_xcd;
    int *h_naive_head, *h_naive_block, *h_naive_xcd;
    int *h_swizzled_head, *h_swizzled_block, *h_swizzled_xcd;

    size = total_wgs * sizeof(int);

    // Allocate / Allokera
    HIP_CHECK(hipMalloc(&d_naive_head, size));
    HIP_CHECK(hipMalloc(&d_naive_block, size));
    HIP_CHECK(hipMalloc(&d_naive_xcd, size));
    HIP_CHECK(hipMalloc(&d_swizzled_head, size));
    HIP_CHECK(hipMalloc(&d_swizzled_block, size));
    HIP_CHECK(hipMalloc(&d_swizzled_xcd, size));

    h_naive_head = (int*)malloc(size);
    h_naive_block = (int*)malloc(size);
    h_naive_xcd = (int*)malloc(size);
    h_swizzled_head = (int*)malloc(size);
    h_swizzled_block = (int*)malloc(size);
    h_swizzled_xcd = (int*)malloc(size);

    // Launch kernel / Starta kernel
    hipLaunchKernelGGL(k_demonstrate_naive_vs_swizzled,
                       dim3(total_wgs), dim3(64), 0, 0,
                       d_naive_head, d_naive_block, d_naive_xcd,
                       d_swizzled_head, d_swizzled_block, d_swizzled_xcd,
                       num_heads, blocks_per_head);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy back / Kopiera tillbaka
    HIP_CHECK(hipMemcpy(h_naive_head, d_naive_head, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_naive_block, d_naive_block, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_naive_xcd, d_naive_xcd, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_swizzled_head, d_swizzled_head, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_swizzled_block, d_swizzled_block, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_swizzled_xcd, d_swizzled_xcd, size, hipMemcpyDeviceToHost));

    printf("  ┌────────────────────────────────┬────────────────────────────────┐\n");
    printf("  │      NAIVE MAPPING             │      SWIZZLED MAPPING          │\n");
    printf("  │      NAIV MAPPNING             │      SWIZZLAD MAPPNING         │\n");
    printf("  ├────────────────────────────────┼────────────────────────────────┤\n");
    printf("  │  WG │ Head │ Block │ XCD      │  WG │ Head │ Block │ XCD       │\n");
    printf("  │─────┼──────┼───────┼──────────│─────┼──────┼───────┼───────────│\n");

    for (int i = 0; i < total_wgs; i++) {
        printf("  │ %3d │  %2d  │   %2d  │   %d      │ %3d │  %2d  │   %2d  │   %d        │\n",
               i, h_naive_head[i], h_naive_block[i], h_naive_xcd[i],
               i, h_swizzled_head[i], h_swizzled_block[i], h_swizzled_xcd[i]);
    }
    printf("  └────────────────────────────────┴────────────────────────────────┘\n");
    printf("\n");

    /**
     * Analyse cache locality / Analysera cache-lokalitet
     */
    printf("  Cache Locality Analysis — Cache-lokalitetsanalys:\n\n");

    printf("  NAIVE: Which XCDs process each head?\n");
    printf("  NAIV: Vilka XCD:er bearbetar varje huvud?\n\n");
    for (int h = 0; h < num_heads; h++) {
        printf("    Head %d: XCDs = {", h);
        for (int b = 0; b < blocks_per_head; b++) {
            int wg = h * blocks_per_head + b;
            printf("%d", h_naive_xcd[wg]);
            if (b < blocks_per_head - 1) printf(", ");
        }
        printf("} — ");
        // Check if all on same XCD / Kontrollera om alla på samma XCD
        bool same_xcd = true;
        for (int b = 1; b < blocks_per_head; b++) {
            int wg0 = h * blocks_per_head;
            int wg = h * blocks_per_head + b;
            if (h_naive_xcd[wg] != h_naive_xcd[wg0]) same_xcd = false;
        }
        printf("%s\n", same_xcd ? "✓ Same XCD (good!)" : "✗ Different XCDs (bad!)");
    }
    printf("\n");

    printf("  SWIZZLED: Which XCDs process each head?\n");
    printf("  SWIZZLAD: Vilka XCD:er bearbetar varje huvud?\n\n");
    for (int h = 0; h < num_heads; h++) {
        printf("    Head %d: XCDs = {", h);
        bool first = true;
        for (int wg = 0; wg < total_wgs; wg++) {
            if (h_swizzled_head[wg] == h) {
                if (!first) printf(", ");
                printf("%d", h_swizzled_xcd[wg]);
                first = false;
            }
        }
        printf("} — ");
        // Check if all on same XCD / Kontrollera om alla på samma XCD
        int first_xcd = -1;
        bool same_xcd = true;
        for (int wg = 0; wg < total_wgs; wg++) {
            if (h_swizzled_head[wg] == h) {
                if (first_xcd < 0) first_xcd = h_swizzled_xcd[wg];
                else if (h_swizzled_xcd[wg] != first_xcd) same_xcd = false;
            }
        }
        printf("%s\n", same_xcd ? "✓ Same XCD (good!)" : "✗ Different XCDs (bad!)");
    }
    printf("\n");

    // Cleanup test 2 / Städa upp test 2
    HIP_CHECK(hipFree(d_naive_head));
    HIP_CHECK(hipFree(d_naive_block));
    HIP_CHECK(hipFree(d_naive_xcd));
    HIP_CHECK(hipFree(d_swizzled_head));
    HIP_CHECK(hipFree(d_swizzled_block));
    HIP_CHECK(hipFree(d_swizzled_xcd));
    free(h_naive_head);
    free(h_naive_block);
    free(h_naive_xcd);
    free(h_swizzled_head);
    free(h_swizzled_block);
    free(h_swizzled_xcd);

    /**
     * ==========================================================================
     * Summary and Next Steps
     * Sammanfattning och nästa steg
     * ==========================================================================
     */
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Summary — Sammanfattning                                                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                              ║\n");
    printf("║  KEY INSIGHT — NYCKELINSIKT:                                                 ║\n");
    printf("║                                                                              ║\n");
    printf("║  1. Naive mapping spreads each head's blocks across ALL XCDs                 ║\n");
    printf("║     Naiv mappning sprider varje huvuds block över ALLA XCD:er                ║\n");
    printf("║     → L2 cache cannot be shared → Low hit rate (1-40%)                       ║\n");
    printf("║     → L2-cache kan inte delas → Låg träfffrekvens (1-40%)                    ║\n");
    printf("║                                                                              ║\n");
    printf("║  2. Swizzled mapping keeps each head's blocks on the SAME XCD                ║\n");
    printf("║     Swizzlad mappning håller varje huvuds block på SAMMA XCD                 ║\n");
    printf("║     → L2 cache IS shared → High hit rate (80-97%)                            ║\n");
    printf("║     → L2-cache DELAS → Hög träfffrekvens (80-97%)                            ║\n");
    printf("║                                                                              ║\n");
    printf("║  3. Performance difference can be up to 50%!                                 ║\n");
    printf("║     Prestandaskillnaden kan vara upp till 50%!                               ║\n");
    printf("║                                                                              ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Next: Run with rocprof to measure actual L2 cache hit rates                 ║\n");
    printf("║  Nästa: Kör med rocprof för att mäta faktiska L2-cache-träfffrekvenser       ║\n");
    printf("║                                                                              ║\n");
    printf("║  Command / Kommando:                                                         ║\n");
    printf("║    rocprof -i metrics.txt ./xcd_discovery                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}