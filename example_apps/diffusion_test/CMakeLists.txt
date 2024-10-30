cmake_minimum_required(VERSION 3.29)

add_executable(${PROJECT_NAME}_diff diff_test.cpp)
target_compile_options(${PROJECT_NAME}_diff PRIVATE -fopenmp)
target_link_libraries(${PROJECT_NAME}_diff Kokkos::kokkos OpenMP::OpenMP_CXX)
