// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which queries the size of tile static area
bool test() {
  // FIXME: use hc namespace
  using namespace Concurrency;

  accelerator acc;
  accelerator_view av = acc.get_default_view();

  // FIXME: migrate to hc::acclerator::get_max_tile_static_size()
  size_t size1 = get_max_tile_static_size(acc);
  std::cout << "Max tile static size of accelerator: " << size1 << "\n";

  // FIXME: migrate to hc::acclerator_view::get_max_tile_static_size()
  size_t size2 = get_max_tile_static_size(av);
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

