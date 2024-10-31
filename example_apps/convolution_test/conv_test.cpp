//
// Created by szylkret on 10/31/24.
//
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <bitmap_image.hpp>

#define CHANNELS 3
#define RED_CHANNEL 0
#define BLUE_CHANNEL 1
#define GREEN_CHANNEL 2

using DualView = Kokkos::DualView<int***, Kokkos::LayoutLeft, Kokkos::Cuda>;

static DualView read_bitmap(std::string path) {
      bitmap_image image(path);
      const std::size_t height = image.height();
      const std::size_t width = image.width();

      DualView image_dual_view("img_data", width, height, CHANNELS);

      auto image_view = image_dual_view.view_host();
      for (std::size_t y = 0; y < height; ++y) {
          for (std::size_t x = 0; x < width; ++x) {
               rgb_t color;
               image.get_pixel(x, y, color);
               image_view(x,y,RED_CHANNEL) = color.red;
               image_view(x,y,GREEN_CHANNEL) = color.green;
               image_view(x,y,BLUE_CHANNEL) = color.blue;
          }
      }

      image_dual_view.modify_host();
      image_dual_view.sync_device();

      return image_dual_view;
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