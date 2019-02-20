// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace hc;

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  int init[vecSize] { 0 };
  for (int i = 0; i < vecSize; ++i) {
    init[i] = i;
  }
  array<int, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](hc::index<1> idx) [[hc]] {
    for(int i = 0; i < vecSize; i++) {
      atomic_fetch_min(&count[i], vecSize / 2);
    }
  });

  array_view<int, 1> av(count);

  bool ret = true;
  // half of the output would be i
  for(int i = 0; i < vecSize / 2; ++i) {
    if(av[i] != i) {
      ret = false;
    }
  }
  // half of the output would be vecSize / 2
  for(int i = vecSize / 2; i < vecSize; ++i) {
    if (av[i] != vecSize / 2) {
      ret = false;
    }
  }

  return !(ret == true);
}
