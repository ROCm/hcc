
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>

using namespace hc;

#define T float
#define TOLERANCE 1e-5
#define INIT 0.5f

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  std::vector<T> init(vecSize, INIT);
  array<T, 1> count(vecSize, init.begin());

  parallel_for_each(count.get_extent(), [=, &count](hc::index<1> idx) [[hc]] {
    for(unsigned i = 0; i < vecSize; i++) {
      atomic_fetch_sub(&count[i], INIT);
    }
  });

  array_view<T, 1> av(count);

  bool ret = true;
  float sum = -std::accumulate(init.begin(), init.end(), 0.0f);
  sum += INIT;
  for(unsigned i = 0; i < vecSize; ++i) {
      if(fabs(av[i] - sum) > TOLERANCE) {
        ret = false;
      }
  }

  return !(ret == true);
}
