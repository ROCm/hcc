
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
  T init[vecSize];
  for (int i = 0; i < vecSize; ++i) {
    init[i] = T(i);
  }
  array<T, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) [[hc]] {
    for(int i = 0; i < vecSize; ++i) {
      atomic_fetch_max(&count[i], T(vecSize / 2));
    }
  });

  array_view<T, 1> av(count);

  bool ret = true;
  // half of the output would be vecSize / 2
  for(int i = 0; i < vecSize / 2; ++i) {
    if (av[i] != T(vecSize / 2)) {
      ret = false;
    }
  }
  // half of the output would be i
  for(int i = vecSize / 2; i < vecSize; ++i) {
    if (av[i] != T(i)) {
      ret = false;
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

