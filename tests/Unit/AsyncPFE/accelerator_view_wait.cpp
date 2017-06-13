
// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

#include <hsa/hsa.h>

#define LOOP_COUNT (10240)

#define TEST_DEBUG (0)

/// test which checks the behavior of:
/// accelerator_view::wait()
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::get_native_handle() to retrieve the
/// underlying hsa_signal_t data structure to query if the kernel has really
/// finished execution after accelerator_view::wait()
///
template<size_t grid_size, size_t tile_size>
hc::completion_future execute(hc::array_view<const int, 1>& av1,
                              hc::array_view<const int, 1>& av2,
                              hc::array_view<int, 1>& av3) {
  // run HC parallel_for_each
  return hc::parallel_for_each(hc::tiled_extent<1>(grid_size, tile_size), [=](hc::tiled_index<1>& idx) restrict(amp) {
    for (int i = 0; i < LOOP_COUNT; ++i) {
      av3(idx) = av1(idx) + av2(idx);
    }
  });
}

template<size_t grid_size>
bool verify(hc::array_view<const int, 1>& av1,
            hc::array_view<const int, 1>& av2,
            hc::array_view<int, 1>& av3) {
  for (int i = 0; i < grid_size; ++i) {
    if (av3[i] != av1[i] + av2[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  bool ret = true;

  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;

  // initialize test data
  std::vector<int> table1(1);
  std::vector<int> table2(1);
  std::vector<int> table3(1);
  for (int i = 0; i < 1; ++i) {
    table1[i] = int_dist(rd);
    table2[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av1(1, table1);
  hc::array_view<const int, 1> av2(1, table2);
  hc::array_view<int, 1> av3(1, table3);

  // launch kernel
  hc::completion_future fut1 = execute<1,1>(av1, av2, av3);

  // obtain native handle
  void* handle1 = fut1.get_native_handle();

  // retrieve HSA signal value
  hsa_signal_value_t signal_value1;
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
#endif


  // initialize test data
  std::vector<int> table4(32);
  std::vector<int> table5(32);
  std::vector<int> table6(32);
  for (int i = 0; i < 32; ++i) {
    table4[i] = int_dist(rd);
    table5[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av4(32, table4);
  hc::array_view<const int, 1> av5(32, table5);
  hc::array_view<int, 1> av6(32, table6);

  // launch kernel
  hc::completion_future fut2 = execute<32,4>(av4, av5, av6);

  // obtain native handle
  void* handle2 = fut2.get_native_handle();

  // retrieve HSA signal value
  hsa_signal_value_t signal_value2;
  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle2));
#if TEST_DEBUG
  std::cout << "signal value #2: " << signal_value2 << "\n";
#endif


  // initialize test data
  std::vector<int> table7(1024);
  std::vector<int> table8(1024);
  std::vector<int> table9(1024);
  for (int i = 0; i < 1024; ++i) {
    table7[i] = int_dist(rd);
    table8[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av7(1024, table7);
  hc::array_view<const int, 1> av8(1024, table8);
  hc::array_view<int, 1> av9(1024, table9);

  // launch kernel
  hc::completion_future fut3 = execute<1024, 16>(av7, av8, av9);

  // obtain native handle
  void* handle3 = fut3.get_native_handle();

  // retrieve HSA signal value
  hsa_signal_value_t signal_value3;
  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle3));
#if TEST_DEBUG
  std::cout << "signal value #3: " << signal_value3 << "\n";
#endif

  // wait on all commands on the default queue to finish
  hc::accelerator().get_default_view().wait();

  // after acclerator_view::wait(), all signals shall become 0 because all
  // kernels are completed
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value1 == 0);

  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle2));
#if TEST_DEBUG
  std::cout << "signal value #2: " << signal_value2 << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value2 == 0);

  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle3));
#if TEST_DEBUG
  std::cout << "signal value #3: " << signal_value3 << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value3 == 0);

  // wait on all commands on the default queue to finish again
  // the signal values should still be 0
  hc::accelerator().get_default_view().wait();

  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value1 == 0);

  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle2));
#if TEST_DEBUG
  std::cout << "signal value #2: " << signal_value2 << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value2 == 0);

  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle3));
#if TEST_DEBUG
  std::cout << "signal value #3: " << signal_value3 << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value3 == 0);

  // verify computation result
  ret &= verify<1>(av1, av2, av3);
  ret &= verify<32>(av4, av5, av6);
  ret &= verify<1024>(av7, av8, av9);

  return !(ret == true);
}

