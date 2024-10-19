cmake_minimum_required(VERSION 3.29)
project(test_kokkos)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)
set(Kokkos_ENABLE_OPENMP ON)
add_subdirectory(third-party/kokkos)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} Kokkos::kokkos OpenMP::OpenMP_C)