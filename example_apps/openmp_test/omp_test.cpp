//
// Created by szylkret on 20.10.2024.
//

#include <iostream>
#include <Kokkos_Core.hpp>

struct add_functor {
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& value_to_add) const {
        value_to_add += i;
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    std::size_t N = 1e5;
    printf("Summation on Kokkos execution space %s\n",
           Kokkos::DefaultExecutionSpace::name());

    double final_value = 0.0;
    Kokkos::parallel_reduce("first reduce", N, add_functor(), final_value);
    printf("Final value: %f\n", final_value);
    Kokkos::finalize();
    return 0;
}
