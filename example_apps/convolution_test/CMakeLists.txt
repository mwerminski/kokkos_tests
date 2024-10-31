

target_compile_options(${PROJECT_NAME}_conv PRIVATE -fopenmp)
target_link_libraries(${PROJECT_NAME}_conv Kokkos::kokkos OpenMP::OpenMP_CXX bitmap_image)
