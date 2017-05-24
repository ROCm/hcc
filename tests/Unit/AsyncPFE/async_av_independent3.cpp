
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (10240)

#define TEST_DEBUG (0)

/// test implicit synchronization of array_view and kernel dispatches
///
/// in this test case, there are NO kernel dependencies because all kernels
/// read from the same read-only array_view instances, and write to DIFFERENT
/// output array_view instances.
template<size_t grid_size, size_t tile_size>
bool test1D() {

  bool ret = true;

  // dependency graph
  // pfe1: av1 + av2 -> av3
  // pfe2: av1 + av2 -> av4
  // pfe3: av1 + av2 -> av5 
  // pfe1, pfe2, pfe3 are all independent

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

#if TEST_DEBUG
  std::cout << "launch pfe1\n";
#endif

  hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av3 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av3(idx) = av1(idx) + av2(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe1\n";
#endif

#if TEST_DEBUG
  std::cout << "launch pfe2\n";
#endif

  // this kernel dispatch shall NOT implicitly wait for the previous one to complete
  // because the array_view written is NOT used by the previous kernels
  hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av4 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av4(idx) = av1(idx) + av2(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe2\n";
#endif

#if TEST_DEBUG
  std::cout << "launch pfe3\n";
#endif

  // this kernel dispatch shall NOT implicitly wait for the previous one to complete
  // because the array_view written is NOT used by the previous kernels
  hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av5 = i * 2
    for (int i = 0; i < LOOP_COUNT; ++i)
      av5(idx) = av1(idx) + av2(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe3\n";
#endif

  // let array_view::synchronize() do implicit wait
#if TEST_DEBUG
  std::cout << "trigger implicit wait on kernels through array_view::synchronize()\n";
#endif

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

