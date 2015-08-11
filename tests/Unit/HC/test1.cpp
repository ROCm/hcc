// XFAIL: Linux
// RUN: %cxxamp %s -Xclang -fhsa-ext -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which checks if the queue is an HSA queue
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  bool ret = true;

  // check if the queue is HSA
  ret &= av.hasHSAInterOp();

  std::cout << ret << "\n";

  // checks if we can get underlying native HSA queue
  void* native_queue = av.getHSAQueue();
  ret &= (native_queue != nullptr);

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

