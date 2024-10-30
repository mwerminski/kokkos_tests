cmake_minimum_required(VERSION 3.29)

add_executable(${PROJECT_NAME}_reduce reduce_test.cpp)
target_compile_options(${PROJECT_NAME}_reduce PRIVATE -fopenmp)
target_link_libraries(${PROJECT_NAME}_reduce Kokkos::kokkos OpenMP::OpenMP_CXX)

