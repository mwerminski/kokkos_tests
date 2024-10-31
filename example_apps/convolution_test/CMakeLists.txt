cmake_minimum_required(VERSION 3.29)

file(COPY example_image.bmp DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
add_executable(${PROJECT_NAME}_conv conv_test.cpp)
target_link_libraries(${PROJECT_NAME}_conv Kokkos::kokkos bitmap_image)
