// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace concurrency;

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  int init[vecSize];
  for (int i = 0; i < vecSize; ++i) {
    init[i] = (i % 2 == 0) ? 0 : 1;
  }
  array<int, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) restrict(amp) {
    // 0 -> 2
    // 1 -> 1
    int v = 0;
    atomic_compare_exchange(&count(idx), &v, 2);
  });

  array_view<int, 1> av(count);

  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
    if (i % 2 == 0) {
      // 0 -> 2
      if (av[i] != 2) {
        ret = false;
      }
    } else {
      // 1 -> 1
      if (av[i] != 1) {
        ret = false;
      }
    }
  }

  return !(ret == true);
}
