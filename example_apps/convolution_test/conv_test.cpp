//
// Created by szylkret on 10/31/24.
//

//#define DIM_SIZE 10000
#define RANK 2

//using HostView = Kokkos::View<double[DIM_SIZE][DIM_SIZE], Kokkos::HostSpace>;

//static HostView read_bitmap(std::string path) {
//  }

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

    // space creation for image
    HostView host_data("host space for img");

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

    }
    Kokkos::finalize();
    return 0;
}