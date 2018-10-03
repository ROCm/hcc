
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

// test new enumeration in hc::accelerator_view : queue_priority
int main() {
  using namespace hc;

  // get the default accelerator
  accelerator acc = accelerator();

  // create one accelerator_view without specifying queue priority
  // by default is shall be priority_normal
  accelerator_view av = acc.create_view();

  // create one accelerator_view with priority normal
  accelerator_view av_normal_priority = acc.create_view(execute_in_order, queuing_mode_automatic, priority_normal);

  // create one accelerator_view with priority high
  accelerator_view av_high_priority = acc.create_view(execute_in_order, queuing_mode_automatic, priority_high);

  // test dispatch a kernel to av
  parallel_for_each(av, extent<1>(1), []() [[hc]] {});

  // test dispatch a kernel to av_normal_priority
  parallel_for_each(av_normal_priority, extent<1>(1), []() [[hc]] {});

  // test dispatch a kernel to av_high_priority
  parallel_for_each(av_high_priority, extent<1>(1), []() [[hc]] {});

  return 0;
}

