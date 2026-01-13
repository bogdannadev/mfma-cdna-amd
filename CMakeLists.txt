cmake_minimum_required(VERSION 3.21)
project(mfma_course LANGUAGES CXX HIP)

set(CMAKE_HIP_ARCHITECTURES gfx942)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")

# Each experiment as separate target
add_executable(hello_hip 01-hello-hip/hello_hip.cpp)
#add_executable(wavefront_basics 02-wavefront-basics/wavefront_basics.cpp)
#add_executable(lds_memory 03-lds-memory/lds_memory.cpp)
#add_executable(mfma_intro 04-mfma-intro/mfma_intro.cpp)
#add_executable(mfma_gemm 05-mfma-gemm/mfma_gemm.cpp)

set_source_files_properties(
        01-hello-hip/hello_hip.cpp
        02-wavefront-basics/wavefront_basics.cpp
        03-lds-memory/lds_memory.cpp
        04-mfma-intro/mfma_intro.cpp
        05-mfma-gemm/mfma_gemm.cpp
        PROPERTIES LANGUAGE HIP
)