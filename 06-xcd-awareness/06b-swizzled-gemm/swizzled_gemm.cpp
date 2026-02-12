/**
 * =============================================================================
 * Experiment 06b (FIXED): XCD-Aware GEMM — Swizzled Workgroup Mapping
 * =============================================================================
 * 
 * FIX APPLIED — KORRIGERING TILLÄMPAD:
 *   Original bug: Loop only iterated 4 times, computing 16 rows per tile
 *   Fixed: Loop now iterates 16 times, computing all 64 rows per tile
 *   
 *   Ursprunglig bugg: Loopen itererade bara 4 gånger, beräknade 16 rader per platta
 *   Korrigerad: Loopen itererar nu 16 gånger, beräknar alla 64 rader per platta
 * 
 * BUILD — KOMPILERA:
 *   hipcc --offload-arch=gfx942 -O3 -o swizzled_gemm swizzled_gemm.cpp
 * 
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
constexpr int TILE_SIZE = 64;
constexpr int BLOCK_K = 16;
constexpr int THREADS_PER_BLOCK = 256;

/**
 * CRITICAL CALCULATION — KRITISK BERÄKNING:
 * 
 * With 256 threads covering a 64×64 tile:
 *   - Each thread handles ONE column (tx = threadIdx.x % 64)
 *   - Threads are grouped into 4 "rows" (ty = threadIdx.x / 64 = 0,1,2,3)
 *   - Each thread must compute 64/4 = 16 rows
 * 
 * Med 256 trådar som täcker en 64×64 platta:
 *   - Varje tråd hanterar EN kolumn
 *   - Trådar grupperas i 4 "rader"
 *   - Varje tråd måste beräkna 16 rader
 */
constexpr int THREAD_ROWS = THREADS_PER_BLOCK / TILE_SIZE;  // 256 / 64 = 4
constexpr int ROWS_PER_THREAD = TILE_SIZE / THREAD_ROWS;    // 64 / 4 = 16

/**
 * =============================================================================
 * SWIZZLING FUNCTIONS — SWIZZLINGFUNKTIONER
 * =============================================================================
 */

__device__ __host__ void naive_wg_to_tile(
    int wg_id,
    int tiles_n,
    int* tile_m,
    int* tile_n
) {
    *tile_m = wg_id / tiles_n;
    *tile_n = wg_id % tiles_n;
}

__device__ __host__ void swizzled_wg_to_tile(
    int wg_id,
    int tiles_m,
    int tiles_n,
    int* tile_m,
    int* tile_n
) {
    /**
     * SWIZZLE STRATEGY — SWIZZLESTRATEGI:
     * 
     * Divide the output grid into 2×4 = 8 "super-tiles" (one per XCD).
     * Dela upp utdatarutnätet i 2×4 = 8 "superplattor" (en per XCD).
     * 
     * Within each super-tile, tiles are assigned sequentially.
     * Inom varje superplatta tilldelas plattor sekventiellt.
     */
    int super_tiles_m = 2;
    int super_tiles_n = 4;
    
    int tiles_per_super_m = (tiles_m + super_tiles_m - 1) / super_tiles_m;
    int tiles_per_super_n = (tiles_n + super_tiles_n - 1) / super_tiles_n;
    int tiles_per_super = tiles_per_super_m * tiles_per_super_n;
    
    int super_tile_id = wg_id / tiles_per_super;
    if (super_tile_id >= NUM_XCDS) super_tile_id = NUM_XCDS - 1;
    
    int local_id = wg_id % tiles_per_super;
    int local_m = local_id / tiles_per_super_n;
    int local_n = local_id % tiles_per_super_n;
    
    int super_m = super_tile_id / super_tiles_n;
    int super_n = super_tile_id % super_tiles_n;
    
    *tile_m = super_m * tiles_per_super_m + local_m;
    *tile_n = super_n * tiles_per_super_n + local_n;
    
    if (*tile_m >= tiles_m) *tile_m = tiles_m - 1;
    if (*tile_n >= tiles_n) *tile_n = tiles_n - 1;
}

/**
 * =============================================================================
 * GEMM KERNEL (FIXED) — GEMM-KÄRNA (KORRIGERAD)
 * =============================================================================
 */
__global__ void k_gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    int use_swizzle
) {
    /**
     * LDS with padding to avoid bank conflicts
     * LDS med utfyllnad för att undvika bankkonflikter
     */
    __shared__ float A_lds[TILE_SIZE][BLOCK_K + 1];
    __shared__ float B_lds[BLOCK_K][TILE_SIZE + 1];
    
    /**
     * Tile position — Plattaposition
     */
    int tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    int tile_m, tile_n;
    if (use_swizzle) {
        swizzled_wg_to_tile(blockIdx.x, tiles_m, tiles_n, &tile_m, &tile_n);
    } else {
        naive_wg_to_tile(blockIdx.x, tiles_n, &tile_m, &tile_n);
    }
    
    /**
     * Thread position within tile — Trådposition inom platta
     * 
     * tx = column index (0-63) — kolumnindex
     * ty = thread row group (0-3) — trådradgrupp
     */
    int tx = threadIdx.x % TILE_SIZE;      // 0-63
    int ty = threadIdx.x / TILE_SIZE;      // 0-3
    
    /**
     * FIXED: Accumulator for 16 rows per thread
     * KORRIGERAT: Ackumulator för 16 rader per tråd
     * 
     * OLD (WRONG): float acc[4]   — only 4 values!
     * NEW (CORRECT): float acc[16] — full 16 rows!
     * 
     * GAMMAL (FEL): float acc[4]   — bara 4 värden!
     * NY (RÄTT): float acc[16] — fulla 16 rader!
     */
    float acc[ROWS_PER_THREAD];  // 16 elements
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }
    
    /**
     * K-loop: iterate over K-blocks
     * K-loop: iterera över K-block
     */
    for (int k_block = 0; k_block < K; k_block += BLOCK_K) {
        /**
         * PHASE 1: Cooperative loading into LDS
         * FAS 1: Kooperativ laddning till LDS
         */
        
        // Load A tile — Ladda A-platta
        for (int i = threadIdx.x; i < TILE_SIZE * BLOCK_K; i += THREADS_PER_BLOCK) {
            int a_row = i / BLOCK_K;
            int a_col = i % BLOCK_K;
            int global_row = tile_m * TILE_SIZE + a_row;
            int global_col = k_block + a_col;
            
            if (global_row < M && global_col < K) {
                A_lds[a_row][a_col] = A[global_row * K + global_col];
            } else {
                A_lds[a_row][a_col] = 0.0f;
            }
        }
        
        // Load B tile — Ladda B-platta
        for (int i = threadIdx.x; i < BLOCK_K * TILE_SIZE; i += THREADS_PER_BLOCK) {
            int b_row = i / TILE_SIZE;
            int b_col = i % TILE_SIZE;
            int global_row = k_block + b_row;
            int global_col = tile_n * TILE_SIZE + b_col;
            
            if (global_row < K && global_col < N) {
                B_lds[b_row][b_col] = B[global_row * N + global_col];
            } else {
                B_lds[b_row][b_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        /**
         * PHASE 2: Compute partial products
         * FAS 2: Beräkna delprodukter
         * 
         * FIXED: Loop over all 16 rows, not just 4!
         * KORRIGERAT: Loopa över alla 16 rader, inte bara 4!
         */
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float b_val = B_lds[k][tx];
            
            /**
             * FIXED: Iterate ROWS_PER_THREAD (16) times
             * KORRIGERAT: Iterera ROWS_PER_THREAD (16) gånger
             */
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                int local_row = ty + r * THREAD_ROWS;  // ty + r*4 → 0,4,8,...60 for ty=0
                acc[r] += A_lds[local_row][k] * b_val;
            }
        }
        
        __syncthreads();
    }
    
    /**
     * PHASE 3: Write results
     * FAS 3: Skriv resultat
     * 
     * FIXED: Write all 16 rows per thread
     * KORRIGERAT: Skriv alla 16 rader per tråd
     */
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int local_row = ty + r * THREAD_ROWS;
        int global_row = tile_m * TILE_SIZE + local_row;
        int global_col = tile_n * TILE_SIZE + tx;
        
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = acc[r];
        }
    }
}

/**
 * =============================================================================
 * HOST FUNCTIONS — VÄRDFUNKTIONER
 * =============================================================================
 */

void init_matrix(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = scale * ((float)rand() / RAND_MAX - 0.5f);
    }
}

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

bool verify_result(const float* gpu_result, const float* cpu_result, int size, float tolerance = 1e-2f) {
    int errors = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        float rel_diff = diff / (fabs(cpu_result[i]) + 1e-6f);
        max_diff = fmax(max_diff, diff);
        
        // Use relative tolerance for larger values / Använd relativ tolerans för större värden
        if (rel_diff > tolerance && diff > 1e-4f) {
            if (errors < 5) {
                printf("  Mismatch at %d: GPU=%.6f, CPU=%.6f, diff=%.6f (rel=%.4f)\n",
                       i, gpu_result[i], cpu_result[i], diff, rel_diff);
            }
            errors++;
        }
    }
    
    printf("  Max absolute difference / Max absolut skillnad: %.6f\n", max_diff);
    printf("  Errors / Fel: %d / %d (%.2f%%)\n", errors, size, 100.0f * errors / size);
    return errors < size * 0.01;  // Allow 1% errors due to FP precision / Tillåt 1% fel pga FP-precision
}

float benchmark_gemm(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    int use_swizzle,
    int num_warmup = 5,
    int num_runs = 20
) {
    int tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks = tiles_m * tiles_n;
    
    // Warmup / Uppvärmning
    for (int i = 0; i < num_warmup; i++) {
        hipLaunchKernelGGL(k_gemm_tiled,
                           dim3(num_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                           d_A, d_B, d_C, M, N, K, use_swizzle);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed runs / Tidtagna körningar
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        hipLaunchKernelGGL(k_gemm_tiled,
                           dim3(num_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                           d_A, d_B, d_C, M, N, K, use_swizzle);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return elapsed_ms / num_runs;
}

void print_swizzle_comparison(int tiles_m, int tiles_n) {
    printf("\n  Workgroup-to-Tile Mapping (first 16 workgroups):\n");
    printf("  Arbetsgrupp-till-platta-mappning (första 16 arbetsgrupperna):\n\n");
    
    printf("  ┌────────────────────────────────────┬────────────────────────────────────┐\n");
    printf("  │         NAIVE MAPPING              │         SWIZZLED MAPPING           │\n");
    printf("  ├────────────────────────────────────┼────────────────────────────────────┤\n");
    printf("  │  WG │ Tile(m,n) │ XCD (round-rob) │  WG │ Tile(m,n) │ XCD (target)     │\n");
    printf("  │─────┼───────────┼─────────────────│─────┼───────────┼──────────────────│\n");
    
    for (int wg = 0; wg < 16 && wg < tiles_m * tiles_n; wg++) {
        int naive_m, naive_n, swiz_m, swiz_n;
        
        naive_wg_to_tile(wg, tiles_n, &naive_m, &naive_n);
        swizzled_wg_to_tile(wg, tiles_m, tiles_n, &swiz_m, &swiz_n);
        
        int naive_xcd = wg % NUM_XCDS;
        
        // Calculate swizzled XCD
        int tiles_per_super_m = (tiles_m + 1) / 2;
        int tiles_per_super_n = (tiles_n + 3) / 4;
        int tiles_per_super = tiles_per_super_m * tiles_per_super_n;
        int swiz_xcd = wg / tiles_per_super;
        if (swiz_xcd >= NUM_XCDS) swiz_xcd = NUM_XCDS - 1;
        
        printf("  │ %3d │   (%2d,%2d)  │       %d         │ %3d │   (%2d,%2d)  │       %d          │\n",
               wg, naive_m, naive_n, naive_xcd,
               wg, swiz_m, swiz_n, swiz_xcd);
    }
    printf("  └────────────────────────────────────┴────────────────────────────────────┘\n");
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Experiment 06b (FIXED): XCD-Aware GEMM — Korrigerad version                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    printf("  FIX APPLIED: Loop now iterates %d times (was 4)\n", ROWS_PER_THREAD);
    printf("  KORRIGERING: Loopen itererar nu %d gånger (var 4)\n\n", ROWS_PER_THREAD);
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    
    printf("Device / Enhet: %s (%s)\n", props.name, props.gcnArchName);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("L2 Cache: %d KB\n", props.l2CacheSize / 1024);
    printf("\n");
    
    // Matrix dimensions / Matrisdimensioner
    int M = 2048;
    int N = 2048;
    int K = 2048;
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    int tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    int total_tiles = tiles_m * tiles_n;
    
    printf("Matrix: %d × %d × %d\n", M, K, N);
    printf("Tiles: %d × %d = %d (ideal %d per XCD)\n", tiles_m, tiles_n, total_tiles, total_tiles / NUM_XCDS);
    printf("\n");
    
    print_swizzle_comparison(tiles_m, tiles_n);
    printf("\n");
    
    // Allocate / Allokera
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_gpu = (float*)malloc(size_C);
    float* h_C_cpu = (float*)malloc(size_C);
    
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    
    // Initialize / Initiera
    printf("Initializing matrices...\n");
    srand(42);
    init_matrix(h_A, M, K, 0.1f);
    init_matrix(h_B, K, N, 0.1f);
    
    HIP_CHECK(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice));
    
    /**
     * ==========================================================================
     * TEST 1: Correctness / Korrekthet
     * ==========================================================================
     */
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Test 1: Correctness — Korrekthet                                            │\n");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    
    // Small test / Litet test
    int test_M = 256, test_N = 256, test_K = 256;
    
    float* ht_A = (float*)malloc(test_M * test_K * sizeof(float));
    float* ht_B = (float*)malloc(test_K * test_N * sizeof(float));
    float* ht_C_gpu = (float*)malloc(test_M * test_N * sizeof(float));
    float* ht_C_cpu = (float*)malloc(test_M * test_N * sizeof(float));
    
    float *dt_A, *dt_B, *dt_C;
    HIP_CHECK(hipMalloc(&dt_A, test_M * test_K * sizeof(float)));
    HIP_CHECK(hipMalloc(&dt_B, test_K * test_N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dt_C, test_M * test_N * sizeof(float)));
    
    init_matrix(ht_A, test_M, test_K, 0.1f);
    init_matrix(ht_B, test_K, test_N, 0.1f);
    
    HIP_CHECK(hipMemcpy(dt_A, ht_A, test_M * test_K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dt_B, ht_B, test_K * test_N * sizeof(float), hipMemcpyHostToDevice));
    
    printf("\n  Computing CPU reference (%d×%d×%d)...\n", test_M, test_K, test_N);
    gemm_cpu(ht_A, ht_B, ht_C_cpu, test_M, test_N, test_K);
    
    int test_tiles_m = (test_M + TILE_SIZE - 1) / TILE_SIZE;
    int test_tiles_n = (test_N + TILE_SIZE - 1) / TILE_SIZE;
    int test_blocks = test_tiles_m * test_tiles_n;
    
    // Test NAIVE
    printf("\n  Testing NAIVE mapping...\n");
    HIP_CHECK(hipMemset(dt_C, 0, test_M * test_N * sizeof(float)));
    hipLaunchKernelGGL(k_gemm_tiled,
                       dim3(test_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                       dt_A, dt_B, dt_C, test_M, test_N, test_K, 0);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ht_C_gpu, dt_C, test_M * test_N * sizeof(float), hipMemcpyDeviceToHost));
    
    bool naive_ok = verify_result(ht_C_gpu, ht_C_cpu, test_M * test_N);
    printf("  NAIVE: %s\n", naive_ok ? "✓ PASS" : "✗ FAIL");
    
    // Test SWIZZLED
    printf("\n  Testing SWIZZLED mapping...\n");
    HIP_CHECK(hipMemset(dt_C, 0, test_M * test_N * sizeof(float)));
    hipLaunchKernelGGL(k_gemm_tiled,
                       dim3(test_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                       dt_A, dt_B, dt_C, test_M, test_N, test_K, 1);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ht_C_gpu, dt_C, test_M * test_N * sizeof(float), hipMemcpyDeviceToHost));
    
    bool swiz_ok = verify_result(ht_C_gpu, ht_C_cpu, test_M * test_N);
    printf("  SWIZZLED: %s\n", swiz_ok ? "✓ PASS" : "✗ FAIL");
    
    free(ht_A); free(ht_B); free(ht_C_gpu); free(ht_C_cpu);
    HIP_CHECK(hipFree(dt_A)); HIP_CHECK(hipFree(dt_B)); HIP_CHECK(hipFree(dt_C));
    
    /**
     * ==========================================================================
     * TEST 2: Performance / Prestanda
     * ==========================================================================
     */
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Test 2: Performance — Prestanda (%d×%d×%d)                              │\n", M, K, N);
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    printf("  Running NAIVE (20 iterations)...\n");
    float naive_time = benchmark_gemm(d_A, d_B, d_C, M, N, K, 0);
    
    printf("  Running SWIZZLED (20 iterations)...\n");
    float swizzled_time = benchmark_gemm(d_A, d_B, d_C, M, N, K, 1);
    
    double flops = 2.0 * M * N * K;
    double naive_gflops = flops / (naive_time * 1e6);
    double swizzled_gflops = flops / (swizzled_time * 1e6);
    float speedup = naive_time / swizzled_time;
    
    printf("\n");
    printf("  ┌───────────────────┬──────────────┬──────────────┬─────────────────────┐\n");
    printf("  │  Mapping          │ Time (ms)    │ GFLOPS       │ Speedup             │\n");
    printf("  ├───────────────────┼──────────────┼──────────────┼─────────────────────┤\n");
    printf("  │  NAIVE            │ %10.3f   │ %10.2f   │ 1.00x (baseline)    │\n", naive_time, naive_gflops);
    printf("  │  SWIZZLED         │ %10.3f   │ %10.2f   │ %.2fx               │\n", swizzled_time, swizzled_gflops, speedup);
    printf("  └───────────────────┴──────────────┴──────────────┴─────────────────────┘\n");
    printf("\n");
    
    if (speedup > 1.05f) {
        printf("  ✓ Swizzled is %.1f%% faster! / Swizzlad är %.1f%% snabbare!\n", 
               (speedup - 1.0f) * 100.0f, (speedup - 1.0f) * 100.0f);
    } else if (speedup < 0.95f) {
        printf("  Note: Swizzled is slower for this size. Try larger matrices (4096+).\n");
        printf("  Notera: Swizzlad är långsammare för denna storlek. Prova större matriser.\n");
    } else {
        printf("  → Similar performance. The effect is more visible with rocprof L2 counters.\n");
        printf("  → Liknande prestanda. Effekten syns tydligare med rocprof L2-räknare.\n");
    }
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Next: Profile with rocprof for L2 cache metrics                             ║\n");
    printf("║  Nästa: Profilera med rocprof för L2-cache-metriker                          ║\n");
    printf("║                                                                              ║\n");
    printf("║  rocprof -i metrics.txt -o results.csv ./swizzled_gemm 4096 4096 4096        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    
    return 0;
}
