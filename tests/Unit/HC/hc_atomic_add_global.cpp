
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
  T init[vecSize] { 0 };
  array<T, 1> count(vecSize, std::begin(init));

  parallel_for_each(count.get_extent(), [=, &count](index<1> idx) [[hc]] {
    for(int i = 0; i < vecSize; i++) {
      atomic_fetch_add(&count[i], T(1));
    }
  });

  array_view<T, 1> av(count);

  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
      if(av[i] != T(vecSize)) {
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

