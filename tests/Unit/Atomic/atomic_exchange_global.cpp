// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace concurrency;

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  int init[vecSize] { 0 };
  array<int, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) restrict(amp) {
    atomic_exchange(&count(idx), 1);
  });

  array_view<int, 1> av(count);

  bool ret = true;
  for(unsigned i = 0; i < vecSize; ++i) {
      if(av[i] != 1) {
        ret = false;
      }
  }

  return !(ret == true);
}
