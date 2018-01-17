
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which checks accelerator::get_hsa_am_region()
// a test which checks accelerator::get_hsa_kernarg_region()
bool test() {

  hc::accelerator acc;

  bool ret = true;

  // check if the device is HSA
  ret &= acc.is_hsa_accelerator();

  std::cout << ret << "\n";

  // checks if we can get AM region
  void* am_region = acc.get_hsa_am_region();
  ret &= (am_region != nullptr);

  std::cout << ret << "\n";

  // checks if we can get Kernarg region
  void* kernarg_region = acc.get_hsa_kernarg_region();
  ret &= (kernarg_region != nullptr);

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

