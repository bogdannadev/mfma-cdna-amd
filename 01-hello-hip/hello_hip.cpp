/**
* =============================================================================
 * Experiment 01: Hello HIP
 * =============================================================================
 *
 * PURPOSE:
 *   Verify HIP compilation and execution on MI300X (gfx942).
 *   Learn the fundamental pattern of GPU programming.
 *
 * KEY CONCEPTS:
 *   - HIP kernel syntax (__global__ functions)
 *   - Thread hierarchy: grid → blocks → threads
 *   - Memory management: hipMalloc, hipMemcpy, hipFree
 *   - Error handling with hipGetLastError()
 *
 * MI300X SPECIFICS (gfx942):
 *   - Wavefront size: 64 threads (vs NVIDIA's 32)
 *   - 304 Compute Units available
 *   - 192 GB VRAM
 *
 * REFERENCE:
 *   HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/en/latest/
 *
 * BUILD:
 *   hipcc --offload-arch=gfx942 -O3 -o hello_hip hello_hip.cpp
 *
 * =============================================================================
 */

#include <hip/hip_runtime.h>
#include <cstdio>

/**
 * Error checking macro - ESSENTIAL for debugging GPU code.
 * GPU errors are often silent; this macro catches them immediately
 */

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

/**
 * GPU Kernel: Runs on the GPU (device), not CPU (host).
 *
 * __global__ marks this as a kernel - callable from host, runs on device.
 *
 * Thread Indexing:
 *   - blockIdx.x  : Which block this thread belongs to (0 to gridDim.x-1)
 *   - blockDim.x  : Number of threads per block
 *   - threadIdx.x : Thread's position within its block (0 to blockDim.x-1)
 *   - Global ID   : blockIdx.x * blockDim.x + threadIdx.x
 *
 * On MI300X, blockDim.x should be multiple of 64 (wavefront size) for efficiency.
 * Threads within a wavefront execute in lockstep (SIMD).
 */

__global__ void vector_square_kernel(float* output, const float* input, int n) {
    // Calculate unique global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // IMPORTANT: Bounds check!
    // We often launch more threads than elements (to fill wavefronts).
    // Extra threads must NOT access the memory
    if (tid < n) {
        output[tid] = input[tid] * input[tid];
    }
}

/**
 *  Helper function to print device properties.
 *  Use this to verify you're on the right GPU.
 */
void print_device_info() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              MI300X Device Information                       ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Device Name:        %-40s ║\n", props.name);
    printf("║ Compute Units:      %-40d ║\n", props.multiProcessorCount);
    printf("║ Wavefront Size:     %-40d ║\n", props.warpSize);
    printf("║ Max Threads/Block:  %-40d ║\n", props.maxThreadsPerBlock);
    printf("║ Global Memory:      %-37.2f GB ║\n", props.totalGlobalMem / 1e9);
    printf("║ Shared Mem/Block:   %-37.2f KB ║\n", props.sharedMemPerBlock / 1024.0);
    printf("║ GCN Architecture:   %-40s ║\n", props.gcnArchName);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

int main() {
    printf("\n=== Experiment 01: Hello HIP ===\n\n");

    // Step 1: Print device info to verify we're on MI300X
    print_device_info();


}