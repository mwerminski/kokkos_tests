cmake_minimum_required(VERSION 3.29)

set(Kokkos_ENABLE_CUDA ON)
set(Kokkos_ARCH_ADA89 ON)

add_executable(${PROJECT_NAME}_cuda cuda_test.cpp)
target_link_libraries(${PROJECT_NAME}_cuda Kokkos::kokkos)