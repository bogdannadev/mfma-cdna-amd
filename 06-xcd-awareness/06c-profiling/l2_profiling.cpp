/**
 * =============================================================================
 * Experiment 06c: L2 Cache Profiling — Demonstrating XCD NUMA Effects
 * =============================================================================
 * 
 * SYFTE (PURPOSE):
 *   Demonstrate L2 cache locality effects using a MEMORY-BOUND kernel
 *   that clearly shows the difference between naive and swizzled mapping.
 *   
 *   Demonstrera L2-cache-lokalitetseffekter med en MINNESBUNDEN kärna
 *   som tydligt visar skillnaden mellan naiv och swizzlad mappning.
 * 
 * WHY A DIFFERENT KERNEL? — VARFÖR EN ANNAN KÄRNA?
 * 
 *   GEMM is compute-bound — each tile does lots of FMA operations per byte loaded.
 *   The L2 benefit is hidden by compute time.
 *   
 *   GEMM är beräkningsbunden — varje platta gör massor av FMA-operationer per byte.
 *   L2-fördelen döljs av beräkningstiden.
 * 
 *   This kernel simulates ATTENTION-LIKE access patterns:
 *   - Multiple "heads" that share KEY data
 *   - Each workgroup reads shared data from its "head"
 *   - Swizzling keeps heads on same XCD → L2 reuse!
 *   
 *   Denna kärna simulerar ATTENTION-LIKNANDE åtkomstmönster:
 *   - Flera "huvuden" som delar KEY-data
 *   - Varje arbetsgrupp läser delad data från sitt "huvud"
 *   - Swizzling håller huvuden på samma XCD → L2-återanvändning!
 * 
 * EXPECTED RESULTS — FÖRVÄNTADE RESULTAT:
 * 
 *   ┌─────────────┬───────────────┬────────────────┐
 *   │ Mapping     │ L2 Hit Rate   │ Performance    │
 *   ├─────────────┼───────────────┼────────────────┤
 *   │ NAIVE       │ 30-50%        │ Slower         │
 *   │ SWIZZLED    │ 80-95%        │ Faster         │
 *   └─────────────┴───────────────┴────────────────┘
 * 
 * BUILD — KOMPILERA:
 *   hipcc --offload-arch=gfx942 -O3 -o l2_profiling l2_profiling.cpp
 * 
 * PROFILE — PROFILERA:
 *   rocprof -i metrics.txt -o results.csv ./l2_profiling
 * 
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/**
 * =============================================================================
 * CONFIGURATION — KONFIGURATION
 * =============================================================================
 */
constexpr int NUM_XCDS = 8;
constexpr int NUM_HEADS = 64;           // Number of "attention heads" / Antal "attention-huvuden"
constexpr int BLOCKS_PER_HEAD = 32;     // Blocks processing each head / Block per huvud
constexpr int KEY_SIZE = 4096;          // Elements per head's KEY data / Element per huvuds KEY-data
constexpr int THREADS_PER_BLOCK = 256;

/**
 * =============================================================================
 * KERNEL: Simulated Attention-like Memory Access
 * =============================================================================
 * 
 * This kernel simulates the memory access pattern of attention:
 * - Each "head" has shared KEY data
 * - Multiple workgroups process the same head
 * - They should all read the same KEY data
 * 
 * Denna kärna simulerar attention-minnesåtkomstmönstret:
 * - Varje "huvud" har delad KEY-data
 * - Flera arbetsgrupper bearbetar samma huvud
 * - De bör alla läsa samma KEY-data
 * 
 * NAIVE: Workgroups for same head spread across XCDs → no L2 sharing
 * SWIZZLED: Workgroups for same head on same XCD → L2 reuse!
 */
__global__ void k_attention_like_access(
    const float* __restrict__ keys,      // [NUM_HEADS][KEY_SIZE]
    float* __restrict__ output,          // [total_workgroups]
    int num_heads,
    int blocks_per_head,
    int key_size,
    int use_swizzle
) {
    int wg_id = blockIdx.x;
    int tid = threadIdx.x;
    
    /**
     * Determine which head this workgroup processes
     * Bestäm vilket huvud denna arbetsgrupp bearbetar
     */
    int head_idx;
    
    if (use_swizzle) {
        /**
         * SWIZZLED: Group heads by XCD
         * SWIZZLAD: Gruppera huvuden per XCD
         * 
         * heads_per_xcd = NUM_HEADS / NUM_XCDS = 64 / 8 = 8
         * blocks_per_xcd = heads_per_xcd * blocks_per_head = 8 * 32 = 256
         * 
         * WG 0-255 → XCD 0, processing heads 0-7
         * WG 256-511 → XCD 1, processing heads 8-15
         * etc.
         */
        int heads_per_xcd = num_heads / NUM_XCDS;
        int blocks_per_xcd = heads_per_xcd * blocks_per_head;
        
        int xcd_id = wg_id / blocks_per_xcd;
        if (xcd_id >= NUM_XCDS) xcd_id = NUM_XCDS - 1;
        
        int local_id = wg_id % blocks_per_xcd;
        int local_head = local_id / blocks_per_head;
        
        head_idx = xcd_id * heads_per_xcd + local_head;
        if (head_idx >= num_heads) head_idx = num_heads - 1;
    } else {
        /**
         * NAIVE: Simple linear mapping
         * NAIV: Enkel linjär mappning
         * 
         * WG 0 → head 0, block 0 → XCD 0
         * WG 1 → head 0, block 1 → XCD 1
         * WG 2 → head 0, block 2 → XCD 2
         * ...
         * WG 32 → head 1, block 0 → XCD 0
         * 
         * Head 0's blocks are on XCDs 0,1,2,3,4,5,6,7,0,1,2,3...
         * NO L2 SHARING between workgroups of same head!
         */
        head_idx = wg_id / blocks_per_head;
        if (head_idx >= num_heads) head_idx = num_heads - 1;
    }
    
    /**
     * Read KEY data for this head
     * Läs KEY-data för detta huvud
     * 
     * ALL workgroups processing the same head read the SAME key data.
     * If they're on the same XCD, this data will be cached in L2!
     * 
     * ALLA arbetsgrupper som bearbetar samma huvud läser SAMMA key-data.
     * Om de är på samma XCD kommer denna data att cachas i L2!
     */
    const float* key_ptr = keys + head_idx * key_size;
    
    /**
     * Each thread reads multiple elements and accumulates
     * Varje tråd läser flera element och ackumulerar
     */
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < key_size; i += THREADS_PER_BLOCK) {
        sum += key_ptr[i];
    }
    
    /**
     * Reduce within workgroup using shared memory
     * Reducera inom arbetsgrupp med delat minne
     */
    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();
    
    // Simple reduction / Enkel reduktion
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[wg_id] = shared[0];
    }
}

/**
 * =============================================================================
 * Benchmark function — Benchmarkfunktion
 * =============================================================================
 */
float benchmark_kernel(
    const float* d_keys,
    float* d_output,
    int num_heads,
    int blocks_per_head,
    int key_size,
    int use_swizzle,
    int num_warmup = 10,
    int num_runs = 50
) {
    int total_blocks = num_heads * blocks_per_head;
    
    // Warmup / Uppvärmning
    for (int i = 0; i < num_warmup; i++) {
        hipLaunchKernelGGL(k_attention_like_access,
                           dim3(total_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                           d_keys, d_output, num_heads, blocks_per_head, key_size, use_swizzle);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed runs / Tidtagna körningar
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        hipLaunchKernelGGL(k_attention_like_access,
                           dim3(total_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                           d_keys, d_output, num_heads, blocks_per_head, key_size, use_swizzle);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return elapsed_ms / num_runs;
}

/**
 * =============================================================================
 * Print XCD distribution — Skriv ut XCD-fördelning
 * =============================================================================
 */
void print_xcd_distribution(int num_heads, int blocks_per_head) {
    printf("\n");
    printf("  XCD Distribution for Head 0 (first 8 workgroups of head 0):\n");
    printf("  XCD-fördelning för huvud 0 (första 8 arbetsgrupperna för huvud 0):\n\n");
    
    printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  NAIVE: Head 0's workgroups spread across ALL XCDs                     │\n");
    printf("  │  NAIV: Huvud 0:s arbetsgrupper sprids över ALLA XCD:er                 │\n");
    printf("  │                                                                         │\n");
    printf("  │    WG 0 (Head 0, Block 0) → XCD 0                                      │\n");
    printf("  │    WG 1 (Head 0, Block 1) → XCD 1                                      │\n");
    printf("  │    WG 2 (Head 0, Block 2) → XCD 2                                      │\n");
    printf("  │    WG 3 (Head 0, Block 3) → XCD 3                                      │\n");
    printf("  │    WG 4 (Head 0, Block 4) → XCD 4                                      │\n");
    printf("  │    WG 5 (Head 0, Block 5) → XCD 5                                      │\n");
    printf("  │    WG 6 (Head 0, Block 6) → XCD 6                                      │\n");
    printf("  │    WG 7 (Head 0, Block 7) → XCD 7  ← All on DIFFERENT XCDs!           │\n");
    printf("  │                                                                         │\n");
    printf("  │  → Each WG loads KEY[0] from HBM independently                         │\n");
    printf("  │  → Varje WG laddar KEY[0] från HBM oberoende av varandra               │\n");
    printf("  │  → L2 cache hit rate: LOW (~30-50%%)                                    │\n");
    printf("  └─────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  SWIZZLED: Head 0's workgroups all on XCD 0                            │\n");
    printf("  │  SWIZZLAD: Huvud 0:s arbetsgrupper alla på XCD 0                       │\n");
    printf("  │                                                                         │\n");
    printf("  │    WG 0 (Head 0, Block 0) → XCD 0                                      │\n");
    printf("  │    WG 1 (Head 0, Block 1) → XCD 0                                      │\n");
    printf("  │    WG 2 (Head 0, Block 2) → XCD 0                                      │\n");
    printf("  │    ...                                                                  │\n");
    printf("  │    WG 31 (Head 0, Block 31) → XCD 0  ← All on SAME XCD!               │\n");
    printf("  │                                                                         │\n");
    printf("  │  → First WG loads KEY[0], others get it from L2 cache!                 │\n");
    printf("  │  → Första WG laddar KEY[0], andra får det från L2-cache!               │\n");
    printf("  │  → L2 cache hit rate: HIGH (~80-95%%)                                   │\n");
    printf("  └─────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Experiment 06c: L2 Cache Profiling — XCD NUMA Effects                       ║\n");
    printf("║  Experiment 06c: L2-cache-profilering — XCD NUMA-effekter                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    
    printf("Device / Enhet: %s (%s)\n", props.name, props.gcnArchName);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("L2 Cache: %d KB (actually 8 × 4MB = 32MB across XCDs)\n", props.l2CacheSize / 1024);
    printf("\n");
    
    /**
     * Configuration / Konfiguration
     */
    int num_heads = NUM_HEADS;
    int blocks_per_head = BLOCKS_PER_HEAD;
    int key_size = KEY_SIZE;
    int total_blocks = num_heads * blocks_per_head;
    
    printf("Configuration / Konfiguration:\n");
    printf("  Heads / Huvuden: %d\n", num_heads);
    printf("  Blocks per head / Block per huvud: %d\n", blocks_per_head);
    printf("  Total workgroups / Totalt arbetsgrupper: %d\n", total_blocks);
    printf("  Key size per head / Key-storlek per huvud: %d floats (%.1f KB)\n", 
           key_size, key_size * sizeof(float) / 1024.0f);
    printf("  Total KEY data / Total KEY-data: %.1f MB\n", 
           (float)num_heads * key_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("  Heads per XCD (swizzled) / Huvuden per XCD: %d\n", num_heads / NUM_XCDS);
    printf("\n");
    
    print_xcd_distribution(num_heads, blocks_per_head);
    
    /**
     * Allocate memory / Allokera minne
     */
    size_t keys_size = num_heads * key_size * sizeof(float);
    size_t output_size = total_blocks * sizeof(float);
    
    float* h_keys = (float*)malloc(keys_size);
    float* h_output = (float*)malloc(output_size);
    
    float *d_keys, *d_output;
    HIP_CHECK(hipMalloc(&d_keys, keys_size));
    HIP_CHECK(hipMalloc(&d_output, output_size));
    
    // Initialize keys / Initiera nycklar
    srand(42);
    for (int i = 0; i < num_heads * key_size; i++) {
        h_keys[i] = (float)rand() / RAND_MAX;
    }
    HIP_CHECK(hipMemcpy(d_keys, h_keys, keys_size, hipMemcpyHostToDevice));
    
    /**
     * ==========================================================================
     * Benchmark both mappings
     * ==========================================================================
     */
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Benchmarking (50 iterations each) — Benchmarkar (50 iterationer vardera)    │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    printf("  Running NAIVE mapping...\n");
    float naive_time = benchmark_kernel(d_keys, d_output, num_heads, blocks_per_head, key_size, 0);
    
    printf("  Running SWIZZLED mapping...\n");
    float swizzled_time = benchmark_kernel(d_keys, d_output, num_heads, blocks_per_head, key_size, 1);
    
    // Calculate bandwidth / Beräkna bandbredd
    // Each workgroup reads key_size floats
    double bytes_read = (double)total_blocks * key_size * sizeof(float);
    double naive_bw = bytes_read / (naive_time * 1e6);      // GB/s
    double swizzled_bw = bytes_read / (swizzled_time * 1e6);
    float speedup = naive_time / swizzled_time;
    
    printf("\n");
    printf("  ┌───────────────────┬──────────────┬──────────────┬─────────────────────┐\n");
    printf("  │  Mapping          │ Time (ms)    │ BW (GB/s)    │ Speedup             │\n");
    printf("  │  Mappning         │ Tid (ms)     │ BW (GB/s)    │ Hastighetsökning    │\n");
    printf("  ├───────────────────┼──────────────┼──────────────┼─────────────────────┤\n");
    printf("  │  NAIVE            │ %10.4f   │ %10.1f   │ 1.00x (baseline)    │\n", naive_time, naive_bw);
    printf("  │  SWIZZLED         │ %10.4f   │ %10.1f   │ %.2fx               │\n", swizzled_time, swizzled_bw, speedup);
    printf("  └───────────────────┴──────────────┴──────────────┴─────────────────────┘\n");
    printf("\n");
    
    if (speedup > 1.1f) {
        printf("  ✓ Swizzled mapping is %.1f%% faster due to better L2 cache utilization!\n",
               (speedup - 1.0f) * 100.0f);
        printf("  ✓ Swizzlad mappning är %.1f%% snabbare tack vare bättre L2-cache-utnyttjande!\n",
               (speedup - 1.0f) * 100.0f);
    } else {
        printf("  → Run with rocprof to see L2 cache hit rate difference:\n");
        printf("  → Kör med rocprof för att se L2-cache-träfffrekvens skillnad:\n");
    }
    
    /**
     * ==========================================================================
     * Summary / Sammanfattning
     * ==========================================================================
     */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Profile Commands — Profileringskommandon                                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                              ║\n");
    printf("║  To measure L2 cache hit rates with rocprof:                                 ║\n");
    printf("║  För att mäta L2-cache-träfffrekvenser med rocprof:                          ║\n");
    printf("║                                                                              ║\n");
    printf("║    rocprof -i metrics.txt -o results.csv ./l2_profiling                      ║\n");
    printf("║                                                                              ║\n");
    printf("║  Then examine results.csv for L2CacheHit values.                             ║\n");
    printf("║  Granska sedan results.csv för L2CacheHit-värden.                            ║\n");
    printf("║                                                                              ║\n");
    printf("║  Expected / Förväntat:                                                       ║\n");
    printf("║    NAIVE kernel:    L2CacheHit ≈ 30-50%%                                      ║\n");
    printf("║    SWIZZLED kernel: L2CacheHit ≈ 80-95%%                                      ║\n");
    printf("║                                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Cleanup / Städa
    free(h_keys);
    free(h_output);
    HIP_CHECK(hipFree(d_keys));
    HIP_CHECK(hipFree(d_output));
    
    return 0;
}
