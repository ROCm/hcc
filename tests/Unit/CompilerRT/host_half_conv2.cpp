// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <hc_defines.h>
#include <iostream>

int main() {
  bool error = false;

  hc::array_view<hc::half,1> av(256);
  hc::parallel_for_each(av.get_extent(), [=](hc::index<1> i) [[hc]] {
    av[i] = static_cast<hc::half>(i[0]+100.0f);
  });

  for (int i = 0; i < av.get_extent()[0] && !error; i++) {
    // test conversion from float to half on the host
    if (std::abs(static_cast<float>(av[i] - static_cast<hc::half>((float)i+100.0f))) >= 0.001f) {
      error = true;
    }
  }
  return error;
}

