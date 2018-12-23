// RUN: %hc %s -lhsa-runtime64 -o %t.out && %t.out

#include <hc/hc.hpp>

#include <iostream>
#include <random>

#include <hsa/hsa.h>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (1024)

#define TEST_DEBUG (0)

// An example which shows how to use accelerator_view::create_blocking_marker(std::initializer_list<completion_future>)
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::get_native_handle() to retrieve the
/// underlying hsa_signal_t data structure to query if dependent kernels have
/// really finished execution before the new kernel is executed.
bool test() {
  bool ret = true;

  // define inputs and output
  const int vecSize = 2048;

  hc::array_view<int, 1> table_a(vecSize);
  hc::array_view<int, 1> table_b(vecSize);
  hc::array_view<int, 1> table_c(vecSize);
  hc::array_view<int, 1> table_d(vecSize);
  hc::array_view<int, 1> table_e(vecSize);
  hc::array_view<int, 1> table_f(vecSize);
  hc::array_view<int, 1> table_g(vecSize);

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    table_a[i] = int_dist(rd);
    table_b[i] = int_dist(rd);
  }

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::completion_future fut0 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_c(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut1 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_d(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut2 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_e(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut3 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_f(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut4 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_g(idx) = table_a(idx) + table_b(idx);
  });

  // create a barrier packet
  hc::accelerator_view av = hc::accelerator().get_default_view();
  hc::completion_future fut5 = av.create_blocking_marker({fut0, fut1, fut2, fut3, fut4}, hc::system_scope);

  // wait on the barrier packet
  fut5.wait();

  // the barrier packet would ensure all previous packets were processed

  // verify
  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += table_c[i] - (table_a[i] + table_b[i]);
    error += table_d[i] - (table_a[i] + table_b[i]);
    error += table_e[i] - (table_a[i] + table_b[i]);
    error += table_f[i] - (table_a[i] + table_b[i]);
    error += table_g[i] - (table_a[i] + table_b[i]);
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

