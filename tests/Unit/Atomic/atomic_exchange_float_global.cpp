// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>

using namespace concurrency;

#define T float
#define INIT 0.5f
#define NEW_VALUE 99.5f

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  std::vector<T> init(vecSize, INIT);
  array<T, 1> count(vecSize, init.begin());

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) restrict(amp) {
    atomic_exchange(&count(idx), NEW_VALUE);
  });

  array_view<T, 1> av(count);

  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
      if(av[i] != NEW_VALUE) {
        ret = false;
      }
  }

  return !(ret == true);
}
