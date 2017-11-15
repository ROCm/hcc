
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>
#include <hc.hpp>

#include <iostream>

/// test HC parallel_for_each interface
template<size_t grid_size>
bool test1D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each
  std::vector<int> table1(grid_size);
  Concurrency::array_view<int, 1> av1(grid_size, table1);

  Concurrency::parallel_for_each(Concurrency::extent<1>(grid_size), [=](Concurrency::index<1>& idx) restrict(amp) {
    av1(idx) = idx[0];
  });


  // next run HC parallel_for_each
  std::vector<int> table5(grid_size);
  hc::array_view<int, 1> av5(grid_size, table5);

  hc::completion_future fut = hc::parallel_for_each(hc::extent<1>(grid_size), [=](hc::index<1>& idx) restrict(amp) {
    av5(idx) = idx[0];
  });

  // wait for kernel to complete
  fut.wait();

#define SHOW_CONTENT_1D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size; ++i) { \
      std::cout << table[i] << " "; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_1D("global", av1, table1)
#endif

#if 0
  SHOW_CONTENT_1D("global", av5, table5)
#endif

#define VERIFY_CONTENT_1D(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT_1D(av1, table1, av5, table5)

  return ret;
}

/// test HC parallel_for_each interface
template<size_t grid_size_0, size_t grid_size_1>
bool test2D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each

  std::vector<int> table1(grid_size_0 * grid_size_1);
  std::vector<int> table2(grid_size_0 * grid_size_1);
  Concurrency::array_view<int, 2> av1(grid_size_0, grid_size_1, table1);
  Concurrency::array_view<int, 2> av2(grid_size_0, grid_size_1, table2);

  Concurrency::parallel_for_each(Concurrency::extent<2>(grid_size_0, grid_size_1), [=](Concurrency::index<2>& idx) restrict(amp) {
    av1(idx) = idx[0];
    av2(idx) = idx[1];
  });

  // next run HC parallel_for_each

  std::vector<int> table9(grid_size_0 * grid_size_1);
  std::vector<int> table10(grid_size_0 * grid_size_1);
  hc::array_view<int, 2> av9(grid_size_0, grid_size_1, table9);
  hc::array_view<int, 2> av10(grid_size_0, grid_size_1, table10);

  hc::completion_future fut = hc::parallel_for_each(hc::extent<2>(grid_size_0, grid_size_1), [=](hc::index<2>& idx) restrict(amp) {
    av9(idx) = idx[0];
    av10(idx) = idx[1];
  });

  // wait for kernel to complete
  fut.wait();

#define SHOW_CONTENT_2D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1; ++i) { \
      std::cout << table[i] << " "; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_2D("global", av1, table1)
  SHOW_CONTENT_2D("global", av2, table2)
#endif

#if 0
  SHOW_CONTENT_2D("global", av9, table9)
  SHOW_CONTENT_2D("global", av10, table10)
#endif

#define VERIFY_CONTENT_2D(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT_2D(av1, table1, av9, table9)
  VERIFY_CONTENT_2D(av2, table2, av10, table10)

  return ret;
}

/// test HC parallel_for_each interface
template<size_t grid_size_0, size_t grid_size_1, size_t grid_size_2>
bool test3D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each

  std::vector<int> table1(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table2(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table3(grid_size_0 * grid_size_1 * grid_size_2);
  Concurrency::array_view<int, 3> av1(grid_size_0, grid_size_1, grid_size_2, table1);
  Concurrency::array_view<int, 3> av2(grid_size_0, grid_size_1, grid_size_2, table2);
  Concurrency::array_view<int, 3> av3(grid_size_0, grid_size_1, grid_size_2, table3);

  Concurrency::parallel_for_each(Concurrency::extent<3>(grid_size_0, grid_size_1, grid_size_2), [=](Concurrency::index<3>& idx) restrict(amp) {
    av1(idx) = idx[0];
    av2(idx) = idx[1];
    av3(idx) = idx[2];
  });

  // next run HC parallel_for_each

  std::vector<int> table13(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table14(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table15(grid_size_0 * grid_size_1 * grid_size_2);
  hc::array_view<int, 3> av13(grid_size_0, grid_size_1, grid_size_2, table13);
  hc::array_view<int, 3> av14(grid_size_0, grid_size_1, grid_size_2, table14);
  hc::array_view<int, 3> av15(grid_size_0, grid_size_1, grid_size_2, table15);

  hc::completion_future fut = hc::parallel_for_each(hc::extent<3>(grid_size_0, grid_size_1, grid_size_2), [=](hc::index<3>& idx) restrict(amp) {
    av13(idx) = idx[0];
    av14(idx) = idx[1];
    av15(idx) = idx[2];
  });

  // wait for kernel to complete
  fut.wait();

#define SHOW_CONTENT_3D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1 * grid_size_2; ++i) { \
      std::cout << table[i] << " "; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_3D("global", av1, table1)
  SHOW_CONTENT_3D("global", av2, table2)
  SHOW_CONTENT_3D("global", av3, table3)
#endif

#if 0
  SHOW_CONTENT_3D("global", av13, table13)
  SHOW_CONTENT_3D("global", av14, table14)
  SHOW_CONTENT_3D("global", av15, table15)
#endif

#define VERIFY_CONTENT_3D(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1 * grid_size_2; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT_3D(av1, table1, av13, table13)
  VERIFY_CONTENT_3D(av2, table2, av14, table14)
  VERIFY_CONTENT_3D(av3, table3, av15, table15)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<1>();
  ret &= test1D<32>();
  ret &= test1D<1024>();
  ret &= test1D<4096>();

  ret &= test2D<1, 1>();
  ret &= test2D<8, 8>();
  ret &= test2D<1024, 16>();
  ret &= test2D<4096, 256>();

  ret &= test3D<1, 1, 1>();
#if 0
  ret &= test3D<8, 8, 8>();
  ret &= test3D<1024, 32, 16>();
#endif

  return !(ret == true);
}

