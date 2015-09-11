// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <iostream>

template<typename T>
bool test() {
  using namespace concurrency;

  int width = 0;

  auto k = [&width] (const index<1>& idx) restrict(amp) {
    width = sizeof(T);
  };

  parallel_for_each(extent<1>(1), k);

  // On HSA, sizeof() result shall agree between CPU and GPU
  return sizeof(T) == width;
}

int main() {
  bool ret = true;

  ret &= test<char>();
  ret &= test<int>();
  ret &= test<unsigned>();
  ret &= test<long>();
  ret &= test<int64_t>();
  ret &= test<float>();
  ret &= test<double>();
  ret &= test<intptr_t>();
  ret &= test<uintptr_t>();

  return !(ret == true);
}

