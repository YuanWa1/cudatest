cmake_minimum_required(VERSION 3.25)

project(cudagpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_hello
        main.cu
        01-vector-add/VectorAdd.cu
)

target_include_directories(cuda_hello PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(cuda_hello PROPERTIES
        CUDA_ARCHITECTURES "89;120"
)