cmake_minimum_required(VERSION 3.29)
add_executable(${PROJECT_NAME}_cuda cuda_test.cpp)
target_link_libraries(${PROJECT_NAME}_cuda Kokkos::kokkos)