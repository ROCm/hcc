
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>

#include <vector>

// test hc::accelerator::get_all_views()
bool test() {
  using namespace hc;
  bool ret = true;

  accelerator acc = accelerator();
  accelerator_view av1 = acc.get_default_view();
  accelerator_view av2 = acc.create_view();
  accelerator_view av3 = acc.create_view();
  accelerator_view av4 = acc.create_view();

  std::vector<accelerator_view> v = acc.get_all_views();

  ret &= (v.size() == 4);
  ret &= (av1 == v[0]);
  ret &= (av2 == v[1]);
  ret &= (av3 == v[2]);
  ret &= (av4 == v[3]);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

