//
// Created by szylkret on 20.10.2024.
//

#include <iostream>
#include <Kokkos_Core.hpp>

#define DIM_SIZE 10000
#define RANK 2

using CudaRangePolicy = Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<RANK>>;
using OpenMPRangePolicy = Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<RANK>>;

using CudaView = Kokkos::View<double[DIM_SIZE][DIM_SIZE], Kokkos::CudaSpace>;
using HostView = Kokkos::View<double[DIM_SIZE][DIM_SIZE], Kokkos::HostSpace>;

template <typename ViewType, typename PolicyType>
static void init_matrix(const double& common_value, const double& point_value, const ViewType& view) {
    const int x_init = DIM_SIZE / 2;
    const int y_init = DIM_SIZE / 2;
    Kokkos::parallel_for("init matrix", PolicyType({0,0},{DIM_SIZE, DIM_SIZE}),
                       KOKKOS_LAMBDA (const int x, const int y) {
                           if (x == x_init and y == y_init) view(x,y) = point_value;
                           else view(x,y) = common_value;
    });
    Kokkos::fence();
}

template <typename ViewType>
struct CalculateDiffFunctor {
    ViewType space_a_;
    ViewType space_b_;
    const double diff_coefficient_; // [m^2/s]
    const double area_; // [m^2]
    const double time_step_; // [s]

    KOKKOS_INLINE_FUNCTION
    void operator()(const int x, const int y) const {
        if (x > 0 and x < DIM_SIZE - 1 and y > 0 and y < DIM_SIZE) {
            space_b_(x,y) += ((-4 * space_a_(x,y))
                                    + space_a_(x-1,y)
                                    + space_a_(x+1,y)
                                    + space_a_(x,y-1)
                                    + space_a_(x,y+1)) * diff_coefficient_ * time_step_ / area_;
        }
    }

    CalculateDiffFunctor(const ViewType& a, const ViewType& b,
                         const double& diff_coeff, const double& area, const double& time_step):
        space_a_(a), space_b_(b), diff_coefficient_(diff_coeff), area_(area), time_step_(time_step) {};
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
      
    // space creation
    CudaView device_data_a("device space a");
    CudaView device_data_b("device space b");
    HostView host_data_a("host space a");
    HostView host_data_b("host space b");

    // diffusion params
    const double diff_coefficient = 1e-10; // [m^2/s]
    const double target_time = 10; // [s]
    const double time_step = 0.000001; // [s]
    const double cell_size = 0.000001; // [m]
    const double area = cell_size * cell_size; // [m^2]

    // matrix init
    init_matrix<CudaView, CudaRangePolicy>(0.16, 2.1, device_data_a);
    Kokkos::deep_copy(device_data_b, device_data_a);
    init_matrix<HostView, OpenMPRangePolicy>(0.16, 2.1, host_data_a);
    Kokkos::deep_copy(host_data_b, host_data_a);

    // functors
    CalculateDiffFunctor<CudaView> cuda_diff_func(device_data_a, device_data_b, diff_coefficient, area, time_step);
    CalculateDiffFunctor<HostView> host_diff_func(host_data_a, host_data_b, diff_coefficient, area, time_step);

    // CUDA execution
    double current_time = 0.0; // [s]
    printf("CUDA execution\n");
    Kokkos::fence();
    Kokkos::Timer timer;
    while (current_time < target_time) {
        Kokkos::parallel_for("cuda_diff_func",
                             CudaRangePolicy({0,0}, {DIM_SIZE,DIM_SIZE}), cuda_diff_func);
        Kokkos::kokkos_swap(device_data_a, device_data_b);
        current_time += time_step;
    }
    Kokkos::fence();
    printf("Time: %f [s]\n", timer.seconds());

    // OpenMP execution
    printf("OpenMP execution\n");
    current_time = 0.0;
    timer.reset();
    while (current_time < target_time) {
        Kokkos::parallel_for("host_diff_func",
                             OpenMPRangePolicy({0,0}, {DIM_SIZE,DIM_SIZE}), host_diff_func);
        Kokkos::kokkos_swap(host_data_a, host_data_b);
        current_time += time_step;
    }
    Kokkos::fence();
    printf("Time: %f [s]\n", timer.seconds());

    }
    Kokkos::finalize();
    return 0;
}
