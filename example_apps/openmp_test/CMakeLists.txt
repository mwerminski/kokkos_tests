cmake_minimum_required(VERSION 3.29)

find_package(OpenMP REQUIRED)
set(Kokkos_ENABLE_OPENMP ON)

add_executable(${PROJECT_NAME}_omp omp_test.cpp)
target_link_libraries(${PROJECT_NAME}_omp Kokkos::kokkos OpenMP::OpenMP_CXX)