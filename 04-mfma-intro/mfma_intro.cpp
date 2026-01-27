/**
 * =============================================================================
 * Experiment 04: MFMA (Matrix Fused Multiply-Add) Introduction
 * =============================================================================
 *
 * PURPOSE:
 *   Understand AMD's Matrix Core instructions - the heart of GPU AI acceleration.
 *   Learn how matrix data is distributed across wavefront lanes.
 *
 * KEY CONCEPTS:
 *   - MFMA computes: D = A × B + C (matrix multiply-accumulate)
 *   - One wavefront (64 threads) executes ONE MFMA instruction
 *   - Matrix elements are distributed across lanes (threads)
 *   - AGPRs (Accumulation GPRs) hold the C/D matrices
 *   - VGPRs hold A and B matrices
 *
 * FROM THE ISA (MI300 ISA PDF, Section 7.1):
 *   "The matrix fused-multiply-add (MFMA) instructions use the matrix core
 *    to perform one or more matrix multiplications."
 *   "The matrix core unit... has the 4×1 by 1×4 outer product as its
 *    fundamental computational primitive."
 *
 * MFMA NAMING CONVENTION:
 *   V_MFMA_[OutputType]_[M]X[N]X[K]_[InputType]
 *   Example: V_MFMA_F32_16X16X16_F16
 *            → 16x16 output (F32), 16 K dimension, F16 inputs
 *
 * COMMON MFMA INSTRUCTIONS FOR LLM INFERENCE:
 *   - v_mfma_f32_16x16x16_f16  : 16×16 output, FP16 inputs (16 cycles)
 *   - v_mfma_f32_32x32x8_f16   : 32×32 output, FP16 inputs (32 cycles)
 *   - v_mfma_f32_16x16x16_bf16 : BF16 variant
 *
 * BUILD:
 *   hipcc --offload-arch=gfx942 -O3 -o mfma_intro mfma_intro.cpp
 *
 * TOOL: Use AMD Matrix Calculator to visualise layouts:
 *   ./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_16x16x16_f16 --detail-instruction
 *
 * =============================================================================
 */

#include <hip/hip_fp16.h> // For __half type
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

/**
 * MFMA VECTOR TYPES:
 *
 * The __builtin_amdgcn_mfma_* intrinsics use LLVM vector types, not HIP types.
 * We define compatible types here.
 */

typedef float v4f32 __attribute__((ext_vector_type(4))); // 4 x float
typedef float v16f32 __attribute((ext_vector_type(16))); // 16 x float
typedef _Float16 v4f16 __attribute__((ext_vector_type(4))); // 4 x half
typedef _Float16 v8f16 __attribute__((ext_vector_type(8))); // 8 x half

#define HIP_CHECK(call)                                                                                                \
    do {                                                                                                               \
        hipError_t err = call;                                                                                         \
        if (err != hipSuccess) {                                                                                       \
            printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)


/**
 * MFMA INTRINSICS:
 *
 * HIP provides built-in functions that map directly to MFMA instructions.
 * The compiler generates the actual v_mfma_* assembly instructions.
 *
 * Format:
 *   __builtin_amdgcn_mfma_f32_MxNxK_f16(A, B, C, cbsz, abid, blgp)
 *
 * Parameters:
 *   A, B: Input matrices (packed in vectors)
 *   C:    Accumulator input (also output location)
 *   cbsz: Control broadcast size (usually 0)
 *   abid: A-matrix broadcast ID (usually 0)
 *   blgp: B-matrix lane group pattern (usually 0)
 *
 * Return: The D matrix (accumulated result)
 */

/**
 * KERNEL 1: Simple MFMA 16x16x16 FP16
 *
 * This is the most common MFMA for LLM inference.
 * - Computes 16x16 output matrix
 * - Uses FP16 inputs, FP32 accumulation
 * - Completes in 16 cycles
 *
 * DATA LAYOUT (crucial to understand!):
 * - 64 lanes in wavefront
 * - 16x16 = 256 output elements
 * - 256 / 64 = 4 elements per lane
 * - Each lane computes 4 elements of the output!
 *
 * The exact mapping of which lane computes which elements is complex.
 * Use the AMD Matrix calculator to visualise it!
 */

__global__ void mfma_16x16x16_f16_kernel(float *D, // Output: 16x16 = 256 floats
                                         const __half *A, // Input A: 16x16
                                         const __half *B, // Input B: 16x16
                                         const float *C) { // Accumulator: 16x16

    // This kernel should be with exactly 64 threads (1 wavefront)
    int lane_id = threadIdx.x;

    /**
     * INPUT DATA LOADING:
     *
     * For v_mfma_f32_16x16x16_f16:
     * - A matrix: Each lane provides 4 FP16 values (v4f16)
     * - B matrix: Each lane provides 4 FP16 values (v4f16)
     * - C matrix: Each lane provides 4 FP16 values (v4f16)
     *
     * Total: 64 lanes x 4 values = 256 values
     */

    // Load A matrix elements for this lane (4 x FP16)
    v4f16 a_vec;
    a_vec[0] = A[lane_id * 4 + 0];
    a_vec[1] = A[lane_id * 4 + 1];
    a_vec[2] = A[lane_id * 4 + 2];
    a_vec[3] = A[lane_id * 4 + 3];

    // Load B matrix elements for this lane (4 x FP16)
    v4f16 b_vec;
    b_vec[0] = B[lane_id * 4 + 0];
    b_vec[1] = B[lane_id * 4 + 1];
    b_vec[2] = B[lane_id * 4 + 2];
    b_vec[3] = B[lane_id * 4 + 3];

    // Load C matrix elements (4 x FP32 accumulator)
    v4f32 c_vec;
    c_vec[0] = C[lane_id * 4 + 0];
    c_vec[1] = C[lane_id * 4 + 1];
    c_vec[2] = C[lane_id * 4 + 2];
    c_vec[3] = C[lane_id * 4 + 3];

    /**
     * EXECUTE MFMA INSTRUCTION!
     *
     * This single intrinsic call generates:
     *   v_mfma_f32_16x16x16_f16 a[0:3], v[A], v[B], a[0:3]
     *
     * Where a[0:3] are Accumulation GPRs (AGPRs)
     *
     * The matrix core performs:
     *  D[16x16] = A[16x16] x B[16x16] + C[16x16]
     *
     * In 16 clock cycles!
     */
    v4f32 d_vec = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, // A matrix data for this lane
                                                        b_vec, // B matrix data fro this lane
                                                        c_vec, // C/D accumulator for this lane
                                                        0, 0, 0 // cbsz, abid, blgp (advanced features, use 0)
    );

    // Store result - each lane writes its 4 output elements
    D[lane_id * 4 + 0] = d_vec[0];
    D[lane_id * 4 + 1] = d_vec[1];
    D[lane_id * 4 + 2] = d_vec[2];
    D[lane_id * 4 + 3] = d_vec[3];
}

/**
 * KERNEL 2: Simpler demonstration using scalar intrinsic
 *
 * This uses a simpler 4x4x4 MFMA to make the concept clearer.
 * Each wavefront computes 16 separate 4x4 matrices!
 *
 * Intrinsic: __builtin_amdgcn_mfma_f32_4x4x1f32
 * - Takes 2 scalar floats (one element of A and B per lane)
 * - 64 lanes / 4 = 16 blocks
 * - Each block of 4 lanes computes one 4x4 outer product
 */
__global__ void mfma_4x4x1_f32_kernel(float *D, const float *A, const float *B, const float *C) {
    int lane_id = threadIdx.x;

    // For v_mfma_f32_4x4x4_16b_f32:
    // Each lane provides 1 element of A and 1 element of B
    // The instruction computes 16 separate 4x4 outer products

    float a_val = A[lane_id];
    float b_val = B[lane_id];

    // C input: 4 values per lane for the 4x4 output
    v4f32 c_vec;
    c_vec.x = C[lane_id * 4 + 0];
    c_vec.y = C[lane_id * 4 + 1];
    c_vec.z = C[lane_id * 4 + 2];
    c_vec.w = C[lane_id * 4 + 3];

    // Execute 4x4x4 MFMA - produces 16 blocks of 4x4 = 256 outputs
    v4f32 d_vec = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, c_vec, 0, 0, 0);

    D[lane_id * 4 + 0] = d_vec[0];
    D[lane_id * 4 + 1] = d_vec[1];
    D[lane_id * 4 + 2] = d_vec[2];
    D[lane_id * 4 + 3] = d_vec[3];
}

/**
 * CPU Reference Implementation for verification
 */
void cpu_matmul_fp32(float *D, const float *A, const float *B, const float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            D[i * N + j] = sum;
        }
    }
}

void print_mfma_info() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           MFMA Instruction Overview (gfx942)                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  Key MFMA instructions for AI/ML:\n\n");
    printf("  ┌─────────────────────────────┬────────┬───────┬──────────┐\n");
    printf("  │ Instruction                 │ Output │ Cycles│ TFLOPS*  │\n");
    printf("  ├─────────────────────────────┼────────┼───────┼──────────┤\n");
    printf("  │ v_mfma_f32_16x16x16_f16     │ 16×16  │  16   │ LLM FP16 │\n");
    printf("  │ v_mfma_f32_32x32x8_f16      │ 32×32  │  32   │ LLM FP16 │\n");
    printf("  │ v_mfma_f32_16x16x16_bf16    │ 16×16  │  16   │ LLM BF16 │\n");
    printf("  │ v_mfma_f32_4x4x1_16b_f32    │ 4×4×16 │   8   │ FP32 demo│\n");
    printf("  │ v_mfma_f64_16x16x4_f64      │ 16×16  │  32   │ FP64 HCP │\n");
    printf("  └─────────────────────────────┴────────┴───────┴──────────┘\n");
    printf("  * TFLOPS depends on clock speed and occupancy\n\n");

    printf("  Register types:\n");
    printf("  ┌──────────┬───────────────────────────────────────────────┐\n");
    printf("  │ VGPR     │ Vector GPRs - inputs A, B matrices           │\n");
    printf("  │ AGPR     │ Accumulation GPRs - C, D matrices            │\n");
    printf("  │ SGPR     │ Scalar GPRs - uniform values (loop counters) │\n");
    printf("  └──────────┴───────────────────────────────────────────────┘\n\n");
}

void test_mfma_concept() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Test: MFMA Conceptual Demo (FP32 4x4x1)              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // For 4x4x4 with 16 blocks: need 64 elements for A, B and 256 for C, D
    const int A_SIZE = 64; // 64 lanes x 1 element
    const int CD_SIZE = 256; // 64 lanes x 4 elements

    float *h_A = (float *) malloc(A_SIZE * sizeof(float));
    float *h_B = (float *) malloc(A_SIZE * sizeof(float));
    float *h_C = (float *) malloc(CD_SIZE * sizeof(float));
    float *h_D = (float *) malloc(CD_SIZE * sizeof(float));

    float *d_A, *d_B, *d_C, *d_D;
    HIP_CHECK(hipMalloc(&d_A, A_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, A_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, A_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_D, A_SIZE * sizeof(float)));

    // Initialize with simple values
    for (int i = 0; i < A_SIZE; i++) {
        h_A[i] = 1.0f; // All 1s
        h_B[i] = 1.0f; // All 1s
    }
    for (int i = 0; i < CD_SIZE; i++) {
        h_C[i] = 0.0f; // Zero accumulator
    }

    HIP_CHECK(hipMemcpy(d_A, h_A, A_SIZE * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, A_SIZE * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, CD_SIZE * sizeof(float), hipMemcpyHostToDevice));

    // Launch with exactly 64 threads (1 wavefront)
    mfma_4x4x1_f32_kernel<<<1, 64>>>(d_D, d_A, d_B, d_C);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_D, d_D, CD_SIZE * sizeof(float), hipMemcpyDeviceToHost));

    printf("  Operation: D = A * B + C (using v_mfma_f32_4x4x1_16b-f32)\n\n");
    printf("  This computes 16 separate 4x4 outer products!\n");
    printf("  Inputs: A = all 1s, B = all 1s, C = all 0s\n");
    printf("  Expected: Each 4x4 block should have value K=4 (dot product of 4 ones)\n\n");

    printf("  First 4x4 output block (lanes 0-3):\n");
    printf("  ┌───────────────────────────┐\n");
    for (int i = 0; i < 4; i++) {
        printf("  │");
        for (int j = 0; j < 4; j++) {
            printf(" %4.1f", h_D[i * 4 + j]);
        }
        printf(" │\n");
    }
    printf("  └───────────────────────────┘\n\n");

    // Check some values
    int sample_correct = (h_D[0] == 4.0f && h_D[15] == 4.0f);
    printf("  Result: %s (expected 4.0 from 1×1×4 dot product)\n\n",
           sample_correct ? "Values look correct ✓" : "Check values ✗");

    printf("  NOTE: The exact output layout is complex!\n");
    printf("  Use: ./matrix_calculator.py --architecture cdna3 \\\n");
    printf("       --instruction v_mfma_f32_4x4x1_16b_f32 --matrix-layout\n\n");

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_D));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
}

void print_register_usage() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         MFMA Register Usage (from ISA PDF)                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  For v_mfma_f32_16x16x16_f16:\n\n");
    printf("  ┌────────────┬────────────────────────────────────────────┐\n");
    printf("  │ Matrix     │ Registers                                  │\n");
    printf("  ├────────────┼────────────────────────────────────────────┤\n");
    printf("  │ A (input)  │ 4 VGPRs (64-bit, holds 4×FP16 per lane)    │\n");
    printf("  │ B (input)  │ 4 VGPRs (64-bit, holds 4×FP16 per lane)    │\n");
    printf("  │ C (accum)  │ 4 AGPRs (holds 4×FP32 per lane)            │\n");
    printf("  │ D (output) │ 4 AGPRs (same as C, in-place)              │\n");
    printf("  └────────────┴────────────────────────────────────────────┘\n\n");

    printf("  Total per wavefront:\n");
    printf("    A: 64 lanes × 4 FP16 = 256 FP16 = 16×16 ✓\n");
    printf("    B: 64 lanes × 4 FP16 = 256 FP16 = 16×16 ✓\n");
    printf("    C: 64 lanes × 4 FP32 = 256 FP32 = 16×16 ✓\n\n");

    printf("  Data movement:\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  Global Memory (HBM)                                    │\n");
    printf("  │         ↓ (global_load)                                 │\n");
    printf("  │  LDS (Local Data Share) - 64KB                          │\n");
    printf("  │         ↓ (ds_read)                                     │\n");
    printf("  │  VGPRs (A, B matrices)                                  │\n");
    printf("  │         ↓ (v_mfma instruction)                          │\n");
    printf("  │  Matrix Core → AGPRs (C/D matrices)                     │\n");
    printf("  │         ↓ (v_accvgpr_read + global_store)               │\n");
    printf("  │  Global Memory (HBM)                                    │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");
}

int main() {
    printf("\n=== Experiment 04: MFMA Introduction ==\n\n");

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    printf("This device %s MFMA instructions.\n\n",
           strstr(props.gcnArchName, "gfx94") ? "SUPPORTS" : "may not support");

    print_mfma_info();
    print_register_usage();
    test_mfma_concept();

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                     Summary                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  ✓ MFMA = Matrix Fused Multiply-Add (D = A×B + C)\n");
    printf("  ✓ One wavefront (64 threads) executes one MFMA\n");
    printf("  ✓ Matrix elements distributed across 64 lanes\n");
    printf("  ✓ AGPRs hold accumulator (C/D), VGPRs hold inputs (A/B)\n");
    printf("  ✓ 16x16x16 FP16 is the sweet spot for LLM inference\n\n");
    printf("  EXERCISES:\n");
    printf("  1. Run matrix_calculator.py to see exact layouts\n");
    printf("  2. Examine assembly: hipcc -S -o mfma_intro.s mfma_intro.cpp\n");
    printf("  3. Look for v_mfma_* instructions in the .s file\n\n");
    printf("NEXT: Experiment 05 - Full MFMA GEMM!\n\n");

    return 0;
}
