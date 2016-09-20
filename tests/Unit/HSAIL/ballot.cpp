
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <cassert>
#include <hc.hpp>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64
#define TEST_DEBUG (0)

int main() {

  hc::array_view<uint64_t,1> a(WAVEFRONT_SIZE);

  hc::extent<1> e(WAVEFRONT_SIZE);

  int errors = 0;
  for (int i = 0; i <  WAVEFRONT_SIZE; i++) {
    hc::parallel_for_each(e,[=](hc::index<1> idx) [[hc]] {
      uint64_t d = hc::__ballot(1);
      if (idx[0]==i)
        a[0] = d;
    }).wait();
    if (a[0] != 0xFFFFFFFFFFFFFFFF) {
       errors++;
    }
#if TEST_DEBUG
    std::cout << "(i=" << i << "): 0x" << std::hex << a[0] << std::endl;
    std::cout << std::dec;
#endif
  }

  return !(errors==0);
}
