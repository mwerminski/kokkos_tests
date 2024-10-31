//
// Created by szylkret on 10/31/24.
//
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <bitmap_image.hpp>

#define RANK 3

using DualView = Kokkos::DualView<double***, Kokkos::LayoutLeft, Kokkos::Cuda>;

static DualView read_bitmap(std::string path) {
      bitmap_image image(path);
      return DualView("img data");
  }

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

    double current_time = 0.0; // [s]
    printf("CUDA execution\n");
    Kokkos::fence();
    Kokkos::Timer timer;

    // something to do .....
    read_bitmap("example_image.bmp");

    Kokkos::fence();
    printf("Time: %f [s]\n", timer.seconds());

    }
    Kokkos::finalize();
    return 0;
}