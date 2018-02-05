
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which checks accelerator_view::get_hsa_agent()
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();
  hc::accelerator_view av2 = acc.create_view();

  bool ret = true;

  // check if the queue is HSA
  ret &= av.is_hsa_accelerator();

  std::cout << ret << "\n";

  // checks if we can get underlying native HSA agent
  void* native_agent = av.get_hsa_agent();
  ret &= (native_agent != nullptr);

  void* native_agent2 = av2.get_hsa_agent();
  ret &= (native_agent2 != nullptr);

  // native_agent and native_agent2 should point to the same agent
  ret &= (native_agent == native_agent2);

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

