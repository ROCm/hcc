
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace hc;

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  int init[vecSize] { 0 };
  array<int, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) [[hc]] {
    for(unsigned i = 0; i < vecSize; i++) {
      atomic_fetch_dec(&count[i]);
    }
  });

  array_view<int, 1> av(count);

  bool ret = true;
  for(unsigned i = 0; i < vecSize; ++i) {
      if(av[i] != -vecSize) {
        ret = false;
      }
  }

  return !(ret == true);
}
