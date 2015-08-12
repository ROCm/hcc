// XFAIL: Linux
// RUN: %cxxamp %s -Xclang -fhsa-ext -I/opt/hsa/include -L/opt/hsa/lib -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

#include <hsa.h>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (1024)

/// test implicit synchronization of array_view and kernel dispatches
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::getNativeHandle() to retrieve the
/// underlying hsa_signal_t data structure to query if dependent kernels have
/// really finished execution before the new kernel is executed.
///
/// in this test case, there are NO kernel dependencies because all kernels
/// read from the same read-only array_view instances, and write to DIFFERENT
/// output array_view instances.
template<size_t grid_size, size_t tile_size>
bool test1D() {

  bool ret = true;

  std::vector<int> table1(grid_size);
  std::vector<int> table2(grid_size);

  std::vector<int> table3(grid_size);
  std::vector<int> table4(grid_size);
  std::vector<int> table5(grid_size);

  for (int i = 0; i < grid_size; ++i) {
    table1[i] = i;
    table2[i] = i;
  }

  hc::array_view<const int, 1> av1(grid_size, table1);
  hc::array_view<const int, 1> av2(grid_size, table2);

  hc::array_view<int, 1> av3(grid_size, table3);
  hc::array_view<int, 1> av4(grid_size, table4);
  hc::array_view<int, 1> av5(grid_size, table5);

  //std::cout << "launch pfe1\n";

  hc::completion_future fut1 = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av3 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av3(idx) = av1(idx) + av2(idx);
  });

  //std::cout << "after pfe1\n";

  void* handle1 = fut1.getNativeHandle();
  hsa_signal_value_t signal_value1;
  signal_value1 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle1));
  //std::cout << "signal value #1: " << signal_value1 << "\n";
  // signal_value1 MUST be 1 because the kernel is just dispatched
  ret &= (signal_value1 == 1);

  //std::cout << "launch pfe2\n";

  // this kernel dispatch shall NOT implicitly wait for the previous one to complete
  // because the array_view written is NOT used by the previous kernels
  hc::completion_future fut2 = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av4 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av4(idx) = av1(idx) + av2(idx);
  });

  //std::cout << "after pfe2\n";

  void* handle2 = fut2.getNativeHandle();
  hsa_signal_value_t signal_value2;
  signal_value1 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle1));
  signal_value2 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle2));
  //std::cout << "signal value #1: " << signal_value1 << "\n";
  //std::cout << "signal value #2: " << signal_value2 << "\n";
  // signal_value1 MUST be 1 because the new kernel has no dependency to it so it's still running
  ret &= (signal_value1 == 1);
  // signal_value2 MUST be 1 because the kernel is just dispatched
  ret &= (signal_value2 == 1);

  //std::cout << "launch pfe3\n";

  // this kernel dispatch shall NOT implicitly wait for the previous one to complete
  // because the array_view written is NOT used by the previous kernels
  hc::completion_future fut3 = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av5 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av5(idx) = av1(idx) + av2(idx);
  });

  //std::cout << "after pfe3\n";

  void* handle3 = fut3.getNativeHandle();
  hsa_signal_value_t signal_value3;
  signal_value1 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle1));
  signal_value2 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle2));
  signal_value3 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle3));
  //std::cout << "signal value #1: " << signal_value1 << "\n";
  //std::cout << "signal value #2: " << signal_value2 << "\n";
  //std::cout << "signal value #3: " << signal_value3 << "\n";
  // signal_value1 MUST be 1 because the new kernel has no dependency to it so it's still running
  ret &= (signal_value1 == 1);
  // signal_value2 MUST be 1 because the new kernel has no dependency to it so it's still running
  ret &= (signal_value2 == 1);
  // signal_value3 MUST be 1 because the kernel is just dispatched
  ret &= (signal_value3 == 1);

  // wait on all kernels to be completed
  hc::accelerator().get_default_view().wait();

  signal_value1 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle1));
  signal_value2 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle2));
  signal_value3 = hsa_signal_load_relaxed(*static_cast<hsa_signal_t*>(handle3));
  //std::cout << "signal value #1: " << signal_value1 << "\n";
  //std::cout << "signal value #2: " << signal_value2 << "\n";
  //std::cout << "signal value #3: " << signal_value3 << "\n";
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
  SHOW_CONTENT_1D("a", av1, table1)
  SHOW_CONTENT_1D("b", av2, table2)
#endif

#if 0
  SHOW_CONTENT_1D("c1", av3, table3)
  SHOW_CONTENT_1D("c2", av4, table4)
  SHOW_CONTENT_1D("c3", av5, table5)
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

  VERIFY_CONTENT_1D(av3, 2);
  VERIFY_CONTENT_1D(av4, 2);
  VERIFY_CONTENT_1D(av5, 2);

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

