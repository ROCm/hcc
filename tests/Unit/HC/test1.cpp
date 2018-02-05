
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which checks if the queue is an HSA queue
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  bool ret = true;

  // check if the queue is HSA
  ret &= av.is_hsa_accelerator();

  std::cout << ret << "\n";

  // checks if we can get underlying native HSA queue
  void* native_queue = av.get_hsa_queue();
  ret &= (native_queue != nullptr);

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

