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
  for (int i = 0; i < vecSize; ++i) {
    init[i] = i;
  }
  array<int, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) restrict(amp) {
    atomic_fetch_xor(&count(idx), 1);
  });

  array_view<int, 1> av(count);

  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
    if ( (i % 2) == 0 ) {
      if (av[i] != (i + 1)) {
        ret = false;
      }
    } else {
      if (av[i] != (i - 1)) {
        ret = false;
      }
    }
  }

  return !(ret == true);
}
