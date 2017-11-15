// RUN: %hc %s -I/opt/rocm/hsa/include -L/opt/rocm/lib -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>

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
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_c(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut1 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_d(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut2 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_e(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut3 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_f(idx) = table_a(idx) + table_b(idx);
  });

  hc::completion_future fut4 = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_g(idx) = table_a(idx) + table_b(idx);
  });

  // create a barrier packet
  hc::accelerator_view av = hc::accelerator().get_default_view();
  hc::completion_future fut5 = av.create_blocking_marker({fut0, fut1, fut2, fut3, fut4}, hc::system_scope);

  void* nativeHandle0 = fut0.get_native_handle();
  void* nativeHandle1 = fut1.get_native_handle();
  void* nativeHandle2 = fut2.get_native_handle();
  void* nativeHandle3 = fut3.get_native_handle();
  void* nativeHandle4 = fut4.get_native_handle();
  void* nativeHandle5 = fut5.get_native_handle();

#if TEST_DEBUG
  std::cout << nativeHandle0 << "\n";
  std::cout << nativeHandle1 << "\n";
  std::cout << nativeHandle2 << "\n";
  std::cout << nativeHandle3 << "\n";
  std::cout << nativeHandle4 << "\n";
  std::cout << nativeHandle5 << "\n";
#endif

  hsa_signal_value_t signal_value0;
  hsa_signal_value_t signal_value1;
  hsa_signal_value_t signal_value2;
  hsa_signal_value_t signal_value3;
  hsa_signal_value_t signal_value4;
  hsa_signal_value_t signal_value5;

  signal_value0 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle0));
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle1));
  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle2));
  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle3));
  signal_value4 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle4));
#if TEST_DEBUG
  std::cout << "kernel signal value: " << signal_value0 << "\n";
  std::cout << "kernel signal value: " << signal_value1 << "\n";
  std::cout << "kernel signal value: " << signal_value2 << "\n";
  std::cout << "kernel signal value: " << signal_value3 << "\n";
  std::cout << "kernel signal value: " << signal_value4 << "\n";
#endif

  signal_value5 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle5));
#if TEST_DEBUG
  std::cout << "blocking barrier signal value: " << signal_value5 << "\n";
#endif

  // wait on the barrier packet
  fut5.wait();

  // the barrier packet would ensure all previous packets were processed

  signal_value0 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle0));
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle1));
  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle2));
  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle3));
  signal_value4 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle4));
#if TEST_DEBUG
  std::cout << "kernel signal value: " << signal_value0 << "\n";
  std::cout << "kernel signal value: " << signal_value1 << "\n";
  std::cout << "kernel signal value: " << signal_value2 << "\n";
  std::cout << "kernel signal value: " << signal_value3 << "\n";
  std::cout << "kernel signal value: " << signal_value4 << "\n";
#endif
  ret &= (signal_value0 == 0);
  ret &= (signal_value1 == 0);
  ret &= (signal_value2 == 0);
  ret &= (signal_value3 == 0);
  ret &= (signal_value4 == 0);

  signal_value5 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle5));
#if TEST_DEBUG
  std::cout << "barrier signal value: " << signal_value5 << "\n";
#endif
  ret &= (signal_value5 == 0);

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

