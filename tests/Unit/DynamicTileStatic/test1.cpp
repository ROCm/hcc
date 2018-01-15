
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which queries the size of tile static area
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  size_t size1 = acc.get_max_tile_static_size();
  std::cout << "Max tile static size of accelerator: " << size1 << "\n";

  size_t size2 = av.get_max_tile_static_size();
  std::cout << "Max tile static size of accelerator_view: " << size2 << "\n";

  // size1 and size2 shall agree
  bool ret = (size1 == size2);
  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

