
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which checks accelerator_view::get_hsa_am_region()
// a test which checks accelerator_view::get_hsa_kernarg_region()
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();
  hc::accelerator_view av2 = acc.create_view();

  bool ret = true;

  // check if the queue is HSA
  ret &= av.is_hsa_accelerator();

  std::cout << ret << "\n";

  // checks if we can get AM region
  void* am_region = av.get_hsa_am_region();
  ret &= (am_region != nullptr);

  void* am_region2 = av2.get_hsa_am_region();
  ret &= (am_region2 != nullptr);

  // am_region and am_region2 should point to the same agent
  ret &= (am_region == am_region2);

  std::cout << ret << "\n";

  // checks if we can get Kernarg region
  void* kernarg_region = av.get_hsa_kernarg_region();
  ret &= (kernarg_region != nullptr);

  void* kernarg_region2 = av2.get_hsa_kernarg_region();
  ret &= (kernarg_region2 != nullptr);

  // kernarg_region and kernarg_region2 should point to the same agent
  ret &= (kernarg_region == kernarg_region2);

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

