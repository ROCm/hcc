
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (10240)

#define TEST_DEBUG (0)

/// test implicit synchronization of array_view and kernel dispatches
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

  // this kernel dispatch shall implicitly wait for the previous one to complete
  // because they access the same array_view instances and write to them
  hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av1 = i * 3
    for (int i = 0; i < LOOP_COUNT; ++i)
      av1(idx) = av2(idx) + av3(idx);
  });

#if TEST_DEBUG
  std::cout << "after pfe2\n";
#endif

#if TEST_DEBUG
  std::cout << "launch pfe3\n";
#endif

  // this kernel dispatch shall implicitly wait for the previous one to complete
  // because they access the same array_view instances and write to them
  hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    // av2 = i * 5
    for (int i = 0; i < LOOP_COUNT; ++i)
      av2(idx) = av1(idx) + av3(idx);
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

