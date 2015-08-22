// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// test accelerator_view::set_av_index() and
//      accelerator_view::get_av_index()
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  bool ret = true;

  // test use set_av_index
  av.set_av_index(999);

  // test use get_av_index;
  ret &= (av.get_av_index() == 999);

  // test cast operator
  ret &= (am_accelerator_view_t(av) == 999);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

