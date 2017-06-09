
// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

#include <hsa/hsa.h>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (10240)

#define TEST_DEBUG (0)

/// test implicit synchronization of array_view and kernel dispatches
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::get_native_handle() to retrieve the
/// underlying hsa_signal_t data structure to query if dependent kernels have
/// really finished execution before the new kernel is executed.
///
template<size_t grid_size, size_t tile_size>
bool test1D() {

  bool ret = true;

  // dependency graph
  // pfe1: av1 + av2 -> av3
  // pfe2: av2 + av3 -> av1
  // pfe3: av3 + av1 -> av2 
  // pfe2 depends on pfe1
  // pfe3 depends on pfe2

  std::vector<int> table1(grid_size);
  std::vector<int> table2(grid_size);
  std::vector<int> table3(grid_size);

  for (int i = 0; i < grid_size; ++i) {
    table1[i] = i;
    table2[i] = i;
  }

  hc::array_view<int, 1> av1(grid_size, table1);
  hc::array_view<int, 1> av2(grid_size, table2);
  hc::array_view<int, 1> av3(grid_size, table3);

#if TEST_DEBUG
  std::cout << "launch pfe1\n";
#endif

  hc::completion_future fut1 = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av3 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av3(idx) = av1(idx) + av2(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe1\n";
#endif

  void* handle1 = fut1.get_native_handle();
  hsa_signal_value_t signal_value1;
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
#endif

#if TEST_DEBUG
  std::cout << "launch pfe2\n";
#endif

  // this kernel dispatch shall implicitly wait for the previous one to complete
  // because they access the same array_view instances and write to them
  hc::completion_future fut2 = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av1 = i * 3
    for (int i = 0; i < LOOP_COUNT; ++i)
      av1(idx) = av2(idx) + av3(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe2\n";
#endif

  void* handle2 = fut2.get_native_handle();
  hsa_signal_value_t signal_value2;
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle2));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
  std::cout << "signal value #2: " << signal_value2 << "\n";
#endif
  // signal_value1 MUST be 0 because the new kernel must wait on the previous one be completed
  ret &= (signal_value1 == 0);

#if TEST_DEBUG
  std::cout << "launch pfe3\n";
#endif

  // this kernel dispatch shall implicitly wait for the previous one to complete
  // because they access the same array_view instances and write to them
  hc::completion_future fut3 = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av2 = i * 5
    for (int i = 0; i < LOOP_COUNT; ++i)
      av2(idx) = av1(idx) + av3(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe3\n";
#endif

  void* handle3 = fut3.get_native_handle();
  hsa_signal_value_t signal_value3;
  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle2));
  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle3));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
  std::cout << "signal value #2: " << signal_value2 << "\n";
  std::cout << "signal value #3: " << signal_value3 << "\n";
#endif
  // signal_value2 MUST be 0 because the new kernel must wait on the previous one be completed
  ret &= (signal_value2 == 0);

  // wait on all kernels to be finished
  hc::accelerator().get_default_view().wait();

  signal_value1 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle1));
  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle2));
  signal_value3 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(handle3));
#if TEST_DEBUG
  std::cout << "signal value #1: " << signal_value1 << "\n";
  std::cout << "signal value #2: " << signal_value2 << "\n";
  std::cout << "signal value #3: " << signal_value3 << "\n";
#endif
  // signal_value1 MUST be 0 because all kernels are finished at this point
  ret &= (signal_value1 == 0);
  // signal_value2 MUST be 0 because all kernels are finished at this point
  ret &= (signal_value2 == 0);
  // signal_value3 MUST be 0 because all kernels are finished at this point
  ret &= (signal_value3 == 0);

#define SHOW_CONTENT_1D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size / tile_size; ++i) { \
      for (int j = 0; j < tile_size; ++j) { \
        std::cout << table[i * tile_size + j] << " "; \
      } \
      std::cout << "\n"; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_1D("av1", av1, table1)
  SHOW_CONTENT_1D("av2", av2, table2)
  SHOW_CONTENT_1D("av3", av3, table3)
#endif

#define VERIFY_CONTENT_1D(av, number) \
  { \
    av.synchronize(); \
    for (int i = 0; i < grid_size; ++i) { \
      if (av[i] != i * number) { \
        ret = false; \
        break; \
      } \
    } \
  } \

  VERIFY_CONTENT_1D(av1, 3);
  VERIFY_CONTENT_1D(av2, 5);
  VERIFY_CONTENT_1D(av3, 2);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<32, 16>();
  ret &= test1D<64, 8>();
  ret &= test1D<128, 32>();
  ret &= test1D<256, 64>();
  ret &= test1D<1024, 256>();

  return !(ret == true);
}

