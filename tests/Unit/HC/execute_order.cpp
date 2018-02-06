
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

// test new enumeration in hc::accelerator_view : execute_order
int main() {
  using namespace hc;

  // get the default accelerator
  accelerator acc = accelerator();

  // create one accelerator_view without specifying execution order
  // by default is shall be in-order
  accelerator_view av = acc.create_view();

  // create one accelerator_view with in-order execution order
  accelerator_view av_in_order = acc.create_view(execute_in_order);

  // create one accelerator_view with any-order execution order
  accelerator_view av_any_order = acc.create_view(execute_any_order);

  // test dispatch a kernel to av
  parallel_for_each(av, extent<1>(1), []() [[hc]] {});

  // test dispatch a kernel to av_in_order
  parallel_for_each(av_in_order, extent<1>(1), []() [[hc]] {});

  // test dispatch a kernel to av_any_order
  parallel_for_each(av_any_order, extent<1>(1), []() [[hc]] {});

  return 0;
}

