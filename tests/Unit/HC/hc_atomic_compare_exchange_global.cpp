
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace hc;

template<typename T>
bool test() {
  const int vecSize = 100;

  // Alloc & init input data
  int init[vecSize];
  for (int i = 0; i < vecSize; ++i) {
    init[i] = (i % 2 == 0) ? T(0) : T(1);
  }
  array<T, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) [[hc]] {
    // 0 -> 2
    // 1 -> 1
    T v = T(0);
    atomic_compare_exchange(&count(idx), &v, T(2));
  });

  array_view<T, 1> av(count);

  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
    if (i % 2 == 0) {
      // 0 -> 2
      if (av[i] != T(2)) {
        ret = false;
      }
    } else {
      // 1 -> 1
      if (av[i] != T(1)) {
        ret = false;
      }
    }
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<unsigned int>();
  ret &= test<int>();
  ret &= test<uint64_t>();

  return !(ret == true);
}

