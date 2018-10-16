
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <iostream>

#define DATA_LEN 7

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

  hc::extent<1> e(DATA_LEN);

  // test dispatch a kernel to av
  hc::array_view<unsigned int, 1> data1(DATA_LEN);
  parallel_for_each(av, e, [=](hc::index<1> i) [[hc]] {
    unsigned int flat_id = i[0];
    data1[flat_id] = flat_id;
  });

  // test dispatch a kernel to av_normal_priority
  hc::array_view<unsigned int, 1> data2(DATA_LEN);
  parallel_for_each(av_normal_priority, e, [=](hc::index<1> i) [[hc]] {
    unsigned int flat_id = i[0];
    data2[flat_id] = flat_id;
  });

  // test dispatch a kernel to av_high_priority
  hc::array_view<unsigned int, 1> data3(DATA_LEN);
  parallel_for_each(av_high_priority, e, [=](hc::index<1> i) [[hc]] {
    unsigned int flat_id = i[0];
    data3[flat_id] = flat_id;
  });

  // verify data
  int errors = 0;
  for (int i = 0; i < DATA_LEN; ++i) {
    if (data1[i] != i) {
      ++errors;
      std::cerr << "Error data1[" << i << "], expected " << i << ", actual " << data1[i] << std::endl;
    }
    if (data2[i] != i) {
      ++errors;
      std::cerr << "Error data2[" << i << "], expected " << i << ", actual " << data2[i] << std::endl;
    }
    if (data3[i] != i) {
      ++errors;
      std::cerr << "Error data3[" << i << "], expected " << i << ", actual " << data3[i] << std::endl;
    }
  }

  return errors==0?0:1;
}

