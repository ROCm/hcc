// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc.hpp>

int main() {

  int test[1] { 0 };
  
  using namespace hc;
  array_view<int, 1> av(1, test);

  parallel_for_each(extent<1>(1), [=](hc::index<1> idx) [[hc]] {
#ifdef __HCC_ACCELERATOR__
    av[idx] = 1;
#else
    av[idx] = 0;
#endif
  });

  av.synchronize();

  return !(test[0] == 1);
}

