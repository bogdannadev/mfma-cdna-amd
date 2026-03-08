#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>

#define HIP_CHECK(call)                                                                          \
    do                                                                                           \
    {                                                                                            \
        hipError_t err = call;                                                                   \
        if(err != hipSuccess)                                                                    \
        {                                                                                        \
            std::fprintf(                                                                        \
                stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while(0)

constexpr uint32_t kWavefrontSize = 64;
constexpr uint32_t kBlockSize     = 256;

constexpr auto div_ceil(auto a, auto b) { return (a + b - 1) / b; }

void gemv_cpu_reference(float* __restrict__ y,
                        const _Float16* __restrict__ A,
                        const _Float16* __restrict__ x,
                        const uint32_t M,
                        const uint32_t K)
{
    for(uint32_t i = 0; i < M; i++)
    {
        float acc = 0.0f;

        for(uint32_t k = 0; k < K; k++)
        {
            acc += static_cast<float>(A[i * K + k]) * static_cast<float>(x[k]);
        }

        y[i] = acc;
    }
}

__global__ void gemv_naive_kernel(float* __restrict__ y,
                                  const _Float16* __restrict__ A,
                                  const _Float16* __restrict__ x,
                                  const uint32_t M,
                                  const uint32_t K)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= M)
        return;

    float acc = 0.0f;

    for(uint32_t k = 0; k < K; k++)
    {
        acc += static_cast<float>(A[row * K + k]) * static_cast<float>(x[k]);
    }

    y[row] = acc;
}

__global__ void gemv_wavefront_kernel(float* __restrict__ y,
                                      const _Float16* __restrict__ A,
                                      const _Float16* __restrict__ x,
                                      const uint32_t M,
                                      const uint32_t K)
{
    // row
    const uint32_t wavefront_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWavefrontSize;
    const uint32_t lane_id      = threadIdx.x % kWavefrontSize;

    if(wavefront_id >= M)
        return;

    float acc = 0.0f;

    for(uint32_t k = lane_id; k < K; k += kWavefrontSize)
    {
        acc += static_cast<float>(A[row * K + k]) * static_cast<float>(x[k]);
    }

    for(uint32_t offset = kWavefrontSize / 2; offset > 0; offset >>= 1)
    {
        acc += __builtin_bit_cast(
            float, __shfl_xor_i32(__builtin_bit_cast(int, acc), offset, kWavefrontSize));
    }

    if(lane_id == 0)
    {
        y[wavefront_id] = acc;
    }
}