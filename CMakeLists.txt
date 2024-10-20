cmake_minimum_required(VERSION 3.29)
project(test_kokkos VERSION 1.0
                    DESCRIPTION "Collection of apps to test Kokkos lib")


set(CMAKE_CXX_STANDARD 23)

add_subdirectory(third-party/kokkos)
add_subdirectory(example_apps/openmp_test)
add_subdirectory(example_apps/cuda_test)
