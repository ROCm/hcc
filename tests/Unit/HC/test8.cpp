
// RUN: %hc %s -o %t.out && %t.out

#include <hc/hc.hpp>

#include <iostream>

// a test which checks accelerator::get_hsa_agent()
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  bool ret = true;

  // check if the queue is HSA
  ret &= av.get_accelerator().is_hsa_accelerator();

  std::cout << ret << "\n";

  // checks if we can get underlying native HSA agent
  void* native_agent = acc.get_hsa_agent();
  ret &= (native_agent != nullptr);

  void* native_agent2 = av.get_accelerator().get_hsa_agent();
  ret &= (native_agent2 != nullptr);

  // native_agent and native_agent2 should point to the same agent
  ret &= static_cast<hsa_agent_t*>(native_agent)->handle ==
    static_cast<hsa_agent_t*>(native_agent2)->handle;

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

