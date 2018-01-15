
// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

#include <hsa/hsa.h>

#define LOOP_COUNT (1024)

#define TEST_DEBUG (0)

/// test which checks the behavior of:
/// completion_future::wait()
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::get_native_handle() to retrieve the
/// underlying hsa_signal_t data structure to query if the kernel has really
/// finished execution after completion_future::wait()
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
  std::vector<int> table1(1024);
  std::vector<int> table2(1024);
  std::vector<int> table3(1024);
  for (int i = 0; i < 1024; ++i) {
    table1[i] = int_dist(rd);
    table2[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av1(1024, table1);
  hc::array_view<const int, 1> av2(1024, table2);
  hc::array_view<int, 1> av3(1024, table3);

  // launch kernel
  hc::completion_future fut = execute<1024, 16>(av1, av2, av3);

  // obtain native handle
  void* handle = fut.get_native_handle();

  // retrieve HSA signal value
  hsa_signal_value_t signal_value;
  signal_value = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle));
#if TEST_DEBUG
  std::cout << "signal value: " << signal_value << "\n";
#endif

  // wait on the future
  fut.wait();

  // after completion_future::wait(), the signal shall become 0 because the
  // kernel is completed
  signal_value = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle));
#if TEST_DEBUG
  std::cout << "signal value: " << signal_value << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value == 0);

  // wait on the future again
  // the signal values should still be 0
  fut.wait();

  signal_value = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle));
#if TEST_DEBUG
  std::cout << "signal value: " << signal_value << "\n";
#endif
  // signal value shall be 0 after the kernel is completed
  ret &= (signal_value == 0);

  // verify computation result
  ret &= verify<1024>(av1, av2, av3);

  return !(ret == true);
}

