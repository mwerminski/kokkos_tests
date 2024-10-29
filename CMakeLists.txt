cmake_minimum_required(VERSION 3.29)
project(test_kokkos VERSION 1.0
                    DESCRIPTION "Collection of apps to test Kokkos lib")


set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP)
set(Kokkos_ENABLE_CUDA ON)
set(Kokkos_ENABLE_CUDA_LAMBDA ON)
set(Kokkos_ENABLE_OPENMP ON)

add_subdirectory(third-party/kokkos)
add_subdirectory(example_apps/reduce_test)
add_subdirectory(example_apps/cuda_test)
