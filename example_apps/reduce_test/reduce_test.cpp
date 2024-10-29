//
// Created by szylkret on 20.10.2024.
//

#include <iostream>
#include <Kokkos_Core.hpp>

typedef Kokkos::RangePolicy<Kokkos::Cuda>   cuda_range_policy;
typedef Kokkos::RangePolicy<Kokkos::OpenMP> openmp_range_policy;

struct add_functor {
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& value_to_add) const {
        value_to_add += i;
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    std::size_t N = 1e5;

    double final_value_1 = 0.0;
    double final_value_2 = 0.0;
    Kokkos::parallel_reduce("reduce_add", cuda_range_policy(0, N), add_functor(), final_value_1);
    Kokkos::parallel_reduce("reduce_add", openmp_range_policy(0, N), add_functor(), final_value_2);

    printf("Final sum CUDA: %f\n", final_value_1);
    printf("Final sum OpenMP: %f\n", final_value_2);
    Kokkos::finalize();
    return 0;
}
