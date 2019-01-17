
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (1024)

#define TEST_DEBUG (0)

// An example which shows how to use completion_future::is_ready()
bool test() {
  bool ret = true;

  // define inputs and output
  const int vecSize = 2048;

  hc::array_view<int, 1> table_a(vecSize);
  hc::array_view<int, 1> table_b(vecSize);
  hc::array_view<int, 1> table_c(vecSize);

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    table_a[i] = int_dist(rd);
    table_b[i] = int_dist(rd);
  }

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::completion_future fut = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i) 
        table_c(idx) = table_a(idx) + table_b(idx);
  });

  // create a barrier packet
  hc::accelerator_view av = hc::accelerator().get_default_view();
  hc::completion_future fut2 = av.create_marker();

  // wait on the barrier packet
  fut2.wait();

  // the barrier packet would ensure all previous packets were processed
  ret &= (fut.is_ready() == true);
  ret &= (fut2.is_ready() == true);

  // verify
  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += table_c[i] - (table_a[i] + table_b[i]);
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  ret &= (error == 0);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}


