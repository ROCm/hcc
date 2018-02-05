// RUN: %cxxamp %s -o %t.out && %t.out

#include <amp.h>

int main() {

  int test[1] { 0 };
  
  using namespace concurrency;
  array_view<int, 1> av(1, test);

  parallel_for_each(extent<1>(1), [=](index<1> idx) restrict(amp) {
#ifdef __HCC_ACCELERATOR__
    av[idx] = 1;
#else
    av[idx] = 0;
#endif
  });

  av.synchronize();

  return !(test[0] == 1);
}

